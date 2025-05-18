import base64
from io import BytesIO
from pathlib import Path
from PIL import Image as PILImage
import torch
import torchvision.transforms.functional as tf
from typing import Union, List


def map_value(
    value: float,
    in_min: float,
    in_max: float,
    out_min: float,
    out_max: float,
) -> float:
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def pil_image_to_tensor(image: PILImage.Image) -> torch.Tensor:
    return tf.pil_to_tensor(image.convert("RGB")).permute(1, 2, 0)


def normalize_tensor_image(image: torch.Tensor) -> torch.Tensor:
    return image / 255.0


def preprocess_image(image: PILImage.Image) -> torch.Tensor:
    tensor = pil_image_to_tensor(image)
    normalized_tensor = normalize_tensor_image(tensor)
    if torch.cuda.is_available():
        normalized_tensor = normalized_tensor.cuda()
    return normalized_tensor


def load_image_as_base64(image: PILImage.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())

    return img_str.decode("utf-8")


def coerce_to_paths(val: Union[str, Path, List[str], List[Path]]) -> List[Path]:

    if isinstance(val, Path):
        return [val]
    if isinstance(val, str):
        return [Path(val)]
    elif isinstance(val, list):
        result = []
        for item in val:
            if isinstance(item, str):
                result.append(Path(item))
            elif isinstance(item, Path):
                result.append(item)
            else:
                raise TypeError(f"Cannot convert to path: {item}")

        return result

    raise TypeError(f"Cannot convert to paths: {val}")
