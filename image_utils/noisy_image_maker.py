from functools import cached_property
from typing import Callable, Optional
from PIL import Image as PILImage
from dataclasses import dataclass, field
import os

from image_utils.image_noiser import ImageNoiser
from image_utils.image_path import ImagePath
from image_utils.noising_operation import NosingOperation
from image_utils.utils import load_image_as_base64


@dataclass
class NoisyImageMaker:
    """
    Generates noisy versions of an image based on a JPEG compression threshold.

    Lazy-loaded properties:
    - `noisy_image`: Created when first accessed.
    - `noisy_base64`: Encoded to base64 when first accessed.
    """

    image_path: ImagePath
    output_path: ImagePath
    noise_operations: list[NosingOperation]
    name: Optional[str] = None

    @classmethod
    def from_str(
        cls,
        image_path: str,
        output_path: str,
        thresholds: list[float],
        noise_fns: list[str],
    ) -> "NoisyImageMaker":
        noise_functions = [getattr(ImageNoiser, name) for name in noise_fns]
        if len(thresholds) != len(noise_functions):
            raise ValueError("Thresholds and noise functions must have the same length")

        noise_operations = [
            NosingOperation(fn, threshold)
            for fn, threshold in zip(noise_functions, thresholds)
        ]

        return cls(
            ImagePath(image_path),
            ImagePath(output_path),
            noise_operations,
        )

    def noisy_image(self) -> PILImage.Image:
        image = self.image_path.load()
        for op in self.noise_operations:
            if op.severity > 0:
                image = op(image, op.severity)
        return image

    def noisy_base64(self) -> str:
        return load_image_as_base64(self.noisy_image())

    def set_fn_threshold(self, noise_fn: Callable, threshold: float):
        for op in self.noise_operations:
            if op.fn == noise_fn:
                op.severity = threshold
                return
        raise ValueError("Noise function not found")

    def __post_init__(self):
        if self.name is None:
            self.name = os.path.basename(self.image_path.path)

        self.validate()

    def validate(self) -> bool:
        """Check if the image is valid."""
        if not isinstance(self.image_path, ImagePath):
            raise TypeError("image_path must be an instance of ImagePath")
