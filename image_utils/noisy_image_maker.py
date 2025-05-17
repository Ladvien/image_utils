from functools import cached_property
from pathlib import Path
from typing import Callable, List, Optional, Tuple
from PIL import Image as PILImage
from dataclasses import dataclass
import os
from random import shuffle

from image_utils.image_noiser import ImageNoiser
from image_utils.image_path import ImagePath
from image_utils.noising_operation import NosingOperation
from image_utils.utils import load_image_as_base64
from labeling.label_manager_config import LabelManagerConfig


@dataclass
class NoisyImageMaker:
    """
    Generates noisy versions of an image with configurable noise operations.

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
        order: int = 1,
    ) -> "NoisyImageMaker":
        noise_functions = [getattr(ImageNoiser, name) for name in noise_fns]
        if len(thresholds) != len(noise_functions):
            raise ValueError("Thresholds and noise functions must match in length")

        noise_operations = [
            NosingOperation(fn, threshold, order)
            for fn, threshold in zip(noise_functions, thresholds)
        ]

        return cls(
            ImagePath(image_path),
            ImagePath(output_path),
            noise_operations,
        )

    @classmethod
    def from_config_shuffled(
        cls,
        image_path: ImagePath,
        output_path: Path,
        noise_and_defaults: List[Tuple[str, float]],
    ) -> "NoisyImageMaker":
        """
        Create a NoisyImageMaker using shuffled noise operation order from config.
        """

        order = list(range(len(noise_and_defaults)))
        shuffle(order)

        noise_operations = [
            NosingOperation.from_str(
                noise_and_defaults[i][0], noise_and_defaults[i][1], i
            )
            for i in order
        ]

        return cls(
            image_path=image_path,
            output_path=output_path,
            noise_operations=noise_operations,
        )

    def update_severity(self, fn_name: str, severity: float):
        """
        Update the severity of a specific noise function.
        """
        for op in self.noise_operations:
            if op.fn.__name__ == fn_name:
                op.severity = severity
                return
        raise ValueError(f"Noise function '{fn_name}' not found.")

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
        if not isinstance(self.image_path, ImagePath):
            raise TypeError("image_path must be an instance of ImagePath")
        return True
