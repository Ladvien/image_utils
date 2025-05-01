from __future__ import annotations
from typing import Callable  # Needed for type hinting
from PIL import Image as PILImage
from random import choice, shuffle, uniform
import numpy as np
from uuid import uuid4
from io import BytesIO


from image_utils.image_loader import ImageLoader
from image_utils.utils import map_value


class ImageNoiser:
    """ """

    corruptions = [
        # I just eyeballed these to see which ones looked good
        "gaussian_noise",
        # "gaussian_blur",
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

        image_paths = list(image_loader.iter_image_paths())
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

            print(f"Saved noisy image: {path}")

    @classmethod
    def with_noise(cls, image: PILImage.Image, severity: float = 0.2) -> PILImage.Image:
        image_to_corrupt = image.copy()
        image_to_corrupt = cls.add_gaussian_noise(image_to_corrupt, severity)
        image_to_corrupt = cls.add_jpeg_compression(image_to_corrupt, severity)

        return image_to_corrupt

    @classmethod
    def add_gaussian_noise(
        cls, image: PILImage.Image, severity: float = 0.2
    ) -> PILImage.Image:
        image_array = np.array(np.array(image)) / 255.0
        image_array = np.clip(
            image_array, np.random.normal(size=image_array.shape, scale=severity), 1.0
        )

        return PILImage.fromarray((image_array * 255).astype(np.uint8))

    @classmethod
    def add_jpeg_compression(
        cls, image: PILImage.Image, severity: float = 0.9
    ) -> PILImage.Image:
        quality = int(map_value(severity, 1, 0, 0, 100))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)

        return PILImage.open(buffer)
