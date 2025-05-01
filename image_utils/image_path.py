from __future__ import annotations
from dataclasses import dataclass
from PIL import Image as PILImage
import os
import base64
from io import BytesIO

from .utils import ImageChecker


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
            print(f"Loading image from path: {self.path}")
            image.show()
        return image

    def load_as_base64(self) -> str:
        buffered = BytesIO()
        image = self.load()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())

        return img_str.decode("utf-8")

    def save(self, path: str) -> None:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        image = self.load()
        image.save(path)
        print(f"Image saved to {path}")

    def is_valid_image(self) -> bool:
        return ImageChecker.is_valid_image(self.path)

    def __post_init__(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Image not found at path: {self.path}")
        if not os.path.isfile(self.path):
            raise ValueError(f"Path is not a file: {self.path}")

        if not self.name:
            self.name = os.path.basename(self.path)
