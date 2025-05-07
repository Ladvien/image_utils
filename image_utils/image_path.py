from __future__ import annotations
from dataclasses import dataclass
from PIL import Image as PILImage
import os
import base64
from io import BytesIO

from .utils import ImageChecker, load_image_as_base64


@dataclass
class ImagePath:
    path: str
    name: str | None = None

    @classmethod
    def from_path(cls, path: str) -> ImagePath:
        return cls(path=path, name=os.path.basename(path))

    def load(self, show_on_load: bool = False) -> PILImage.Image:
        image = PILImage.open(self.path).convert("RGB")
        if show_on_load:
            image.show()
        return image

    def load_as_base64(self) -> str:
        image = self.load()
        return load_image_as_base64(image)

    def save(self, path: str) -> None:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        image = self.load()
        image.save(path)
        print(f"Image saved to {path}")

    def is_valid_image(self) -> bool:
        return ImageChecker.is_valid_image(self.path)

    def __post_init__(self):
        if not self.name:
            self.name = os.path.basename(self.path)
