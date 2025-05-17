from typing import Callable
from PIL import Image as PILImage
from dataclasses import dataclass, asdict

from image_utils.image_noiser import ImageNoiser


@dataclass
class NosingOperation:
    fn: Callable[[PILImage.Image, float], PILImage.Image]
    severity: float
    order: int
    name: str | None = None

    @classmethod
    def from_str(cls, name: str, severity: float, order: int) -> "NosingOperation":
        fn = getattr(ImageNoiser, name)
        return cls(fn, severity, order)

    def __call__(self, image: PILImage.Image, severity: float) -> PILImage.Image:
        return self.fn(image, severity)

    def to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        return f"NosingOperation(name={self.name}, severity={self.severity}, order={self.order})"

    def __post_init__(self):
        if self.name is None:
            self.name = self.fn.__name__
        if not callable(self.fn):
            raise ValueError(f"Function {self.fn} is not callable.")
        if not isinstance(self.severity, float):
            raise ValueError(f"Severity {self.severity} is not a float.")
