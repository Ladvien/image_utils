import pytest
from pathlib import Path
from PIL import Image as PILImage
import shutil
import tempfile
import torch

from image_utils.utils import (
    ImageChecker,
    map_value,
    pil_image_to_tensor,
    normalize_tensor_image,
    preprocess_image,
)


def test_is_valid_image_true(tmp_image_file):
    assert ImageChecker.is_valid_image(str(tmp_image_file)) == True


def test_is_valid_image_false(tmp_non_image_file):
    assert ImageChecker.is_valid_image(str(tmp_non_image_file)) == False


def test_is_valid_image_directory(tmp_image_file):
    assert ImageChecker.is_valid_image(str(tmp_image_file.parent)) == False


def test_map_value_scaling():
    assert map_value(5, 0, 10, 0, 100) == 50
    assert map_value(0, 0, 10, -1, 1) == -1
    assert map_value(10, 0, 10, -1, 1) == 1


def test_pil_image_to_tensor(tmp_image_file):
    img = PILImage.open(tmp_image_file)
    tensor = pil_image_to_tensor(img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape[-1] == 3  # RGB channels


def test_normalize_tensor_image(tmp_image_file):
    img = PILImage.open(tmp_image_file)
    tensor = pil_image_to_tensor(img)
    normalized = normalize_tensor_image(tensor)
    assert isinstance(normalized, torch.Tensor)
    assert torch.all(normalized >= 0) and torch.all(normalized <= 1)


def test_preprocess_image(tmp_image_file):
    img = PILImage.open(tmp_image_file)
    processed = preprocess_image(img)
    assert isinstance(processed, torch.Tensor)
    assert processed.shape[-1] == 3
    assert torch.all(processed >= 0) and torch.all(processed <= 1)
