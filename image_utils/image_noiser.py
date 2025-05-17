from __future__ import annotations
from typing import Callable  # Needed for type hinting
from PIL import Image as PILImage
from PIL import ImageFilter, ImageEnhance
from random import choice, shuffle, uniform
import numpy as np
from uuid import uuid4
from io import BytesIO
from random import sample, random
from typing import Tuple, Dict

from image_utils.image_loader import ImageLoader
from image_utils.utils import map_value


class ImageNoiser:
    """ """

    corruptions = [
        # I just eyeballed these to see which ones looked good
        "gaussian_noise",
        "gaussian_blur",
        "jpeg_compression",
        # "contrast",
        # "elastic_transform",
        # "shot_noise",
        # "impulse_noise",
        # "defocus_blur",
        # "glass_blur",
        # "motion_blur",
        # "zoom_blur",
        # "snow",
        # "frost",
        # "fog",
        # "brightness",
        # "pixelate",
        # "speckle_noise",
        # "spatter",
        # "saturate",
    ]

    @classmethod
    def noise_images(
        cls,
        image_loader: ImageLoader,
        output_folder: str,
        severity_range: tuple[float, float],
        noise_functions=list[Callable],
        samples: int | None = None,
    ) -> None:
        """
        Adds noise to images in the specified folder.

        Args:
            image_loader (ImageLoader): The image loader object.
            output_folder (str): The folder to save the noisy images.
            severity (float): The severity of the noise.
            noise_functions (list[Callable]): List of noise functions to apply.
            samples (int | None): Number of samples to process. If None, process all images.
        """

        if samples is None:
            samples = len(image_loader.image_paths)

        image_paths = list(image_loader)
        shuffle(image_paths)

        for i, image_path in enumerate(image_paths):
            if i >= samples:
                break

            image = image_path.load()
            noise_function = choice(noise_functions)
            severity_min, severity_max = severity_range
            severity = uniform(severity_min, severity_max)
            noisy_image: PILImage.Image = noise_function(image, severity)

            unique_id = str(uuid4())
            path = f"{output_folder}/{unique_id}_{image_path.name}_noisy.jpg"
            noisy_image.save(path, quality=95)

    @classmethod
    def with_noise(cls, image: PILImage.Image, severity: float = 0.2) -> PILImage.Image:
        image_to_corrupt = image.copy()
        image_to_corrupt = cls.add_gaussian_noise(image_to_corrupt, severity)
        image_to_corrupt = cls.add_jpeg_compression(image_to_corrupt, severity)

        return image_to_corrupt

    @classmethod
    def add_gaussian_noise(
        cls, image: PILImage.Image, severity: float
    ) -> PILImage.Image:
        print(severity)
        sigma = map_value(severity, 0, 1, 0, 0.3)
        image_array = np.array(image).astype(np.float32) / 255.0
        noise = np.random.normal(0, sigma, image_array.shape)
        noisy_image = image_array + noise
        noisy_image = np.clip(noisy_image, 0, 1)
        return PILImage.fromarray((noisy_image * 255).astype(np.uint8))

    @classmethod
    def add_jpeg_compression(
        cls, image: PILImage.Image, severity: float
    ) -> PILImage.Image:
        quality = int(map_value(severity, 1, 0, 0, 100))
        buffer = BytesIO()
        if image.mode != "RGB":
            image = image.convert("RGB")

        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)

        return PILImage.open(buffer)

    @classmethod
    def add_gaussian_blur(
        cls, image: PILImage.Image, severity: float
    ) -> PILImage.Image:

        # Map severity to a blur radius (adjust max to taste)
        radius = map_value(severity, 0, 1, 0, 10)
        blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
        return blurred

    @classmethod
    def add_brightness(cls, image: PILImage.Image, severity: float) -> PILImage.Image:
        factor = map_value(severity, 0, 1, 1.0, 2.0)  # darker to brighter
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    @classmethod
    def add_high_contrast(
        cls, image: PILImage.Image, severity: float
    ) -> PILImage.Image:
        factor = map_value(severity, 0, 1, 1.0, 2.0)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    @classmethod
    def add_low_contrast(cls, image: PILImage.Image, severity: float) -> PILImage.Image:
        factor = map_value(severity, 0, 1, 1.0, 0.0)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    @classmethod
    def add_saturation(cls, image: PILImage.Image, severity: float) -> PILImage.Image:
        factor = map_value(severity, 0, 1, 1.0, 0.0)
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)

    @classmethod
    def add_pixelate(cls, image: PILImage.Image, severity: float) -> PILImage.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")

        factor = int(map_value(severity, 0, 1, 1, 20))
        w, h = image.size
        image = image.resize(
            (max(1, w // factor), max(1, h // factor)), PILImage.NEAREST
        )
        return image.resize((w, h), PILImage.NEAREST)

    @classmethod
    def add_all_noises(
        cls, image: PILImage.Image, severity: float, num_noise_fns: int | None = None
    ) -> Tuple[PILImage.Image, Dict[str, float]]:
        """
        Blend multiple noise types into one image, with no leftover original.

        Args:
            image:         original PIL image
            severity:      total noise budget (0.0 = none, max = number of fns)
            num_noise_fns: how many distinct noises to mix (defaults to all)

        Returns:
            (blended_image, {fn_name: weight, …})

        Raises:
            ValueError: if severity > num_noise_fns
        """
        # 1. pick noise functions
        funcs = [getattr(cls, f"add_{name}") for name in cls.corruptions]
        n = len(funcs) if num_noise_fns is None else min(num_noise_fns, len(funcs))

        if severity < 0:
            raise ValueError(f"Severity must be >= 0, got {severity}")
        if severity > n:
            raise ValueError(
                f"Severity {severity:.3f} exceeds max total {n} " f"(one per noise fn)."
            )

        chosen = sample(funcs, n)

        # 2. randomly split the total severity across them
        raw = [random() for _ in range(n)]
        total_raw = sum(raw)
        weights = [r / total_raw * severity for r in raw]

        # 3. accumulate only noise contributions
        orig_arr = np.array(image).astype(np.float32)
        result = np.zeros_like(orig_arr)  # start at zero (no original)

        applied: Dict[str, float] = {}
        for fn, w in zip(chosen, weights):
            # get the maximal corruption (severity=1.0)
            full_img = fn(image.copy(), 1.0)
            full_arr = np.array(full_img).astype(np.float32)
            result += full_arr * w
            applied[fn.__name__] = w

        # 4. finalize blend and return
        result = np.clip(result, 0, 255).astype(np.uint8)
        blended = PILImage.fromarray(result)

        return blended, applied


if __name__ == "__main__":
    from pathlib import Path
    import matplotlib.pyplot as plt

    from image_utils.image_loader import ImageLoader
    from image_utils.image_noiser import ImageNoiser

    # Configuration
    image_dir = Path("/Volumes/Shared/external_ssd/datasets/laion_aesthetics/45/05")
    severity = 1.5
    num_noise_fns = 3
    num_images_to_display = 5

    # Load images
    loader = ImageLoader(str(image_dir), shuffle=True)
    assert len(loader) > 0, f"No valid images found in {image_dir}"

    # Loop through images and display side-by-side
    for i, img_path in enumerate(loader):
        if i >= num_images_to_display:
            break

        original = img_path.load()
        blended, weights = ImageNoiser.add_all_noises(original, severity, num_noise_fns)

        # Plot original and noised
        plt.figure(figsize=(12, 5))
        plt.suptitle(f"{img_path.name}", fontsize=10)

        plt.subplot(1, 2, 1)
        plt.imshow(original)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(blended)
        plt.title(
            f"Noised (severity={severity}, n={num_noise_fns})\n"
            + "\n".join(f"{k}: {v:.2f}" for k, v in weights.items())
        )
        plt.axis("off")

        plt.tight_layout()
        plt.show()
