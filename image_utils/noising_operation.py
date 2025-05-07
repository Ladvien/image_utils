from typing import Callable
from PIL import Image as PILImage
from dataclasses import dataclass, asdict

from image_utils.image_noiser import ImageNoiser


@dataclass
class NosingOperation:
    def __init__(
        self,
        fn: Callable[[PILImage.Image, float], PILImage.Image],
        severity: float,
    ):
        self.name = fn.__name__
        self.fn = fn
        self.severity = severity

    @classmethod
    def from_str(cls, name: str, severity: float) -> "NosingOperation":
        fn = getattr(ImageNoiser, name)
        return cls(fn, severity)

    def __call__(self, image: PILImage.Image, severity: float) -> PILImage.Image:
        return self.fn(image, severity)

    def to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        return f"NosingOperation(name={self.name}, severity={self.severity})"
