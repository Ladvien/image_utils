import os
import pytest
from pathlib import Path
from PIL import Image as PILImage
import shutil
import tempfile

from image_utils.image_loader import ImageLoader
from image_utils.image_path import ImagePath


def test_loader_finds_images(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    assert len(loader) == 3


def test_loader_iterates_images(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    paths = list(loader)
    assert all(isinstance(p, ImagePath) for p in paths)
    assert len(paths) == 3


def test_loader_iterates_loaded_images(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    images = list(loader.images())
    assert all(isinstance(img, PILImage.Image) for img in images)
    assert len(images) == 3


def test_loader_indexing(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    img_path = loader[0]
    assert isinstance(img_path, ImagePath)


def test_loader_len(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    assert len(loader) == 3


def test_loader_get_all_image_paths(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    paths = loader.all_image_paths()
    assert len(paths) == 3
    assert all(isinstance(p, Path) or isinstance(p, str) for p in paths)


def test_loader_get_all_images(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    images = loader.all_images()
    assert len(images) == 3
    assert all(isinstance(img, PILImage.Image) for img in images)


def test_loader_get_image(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    img = loader.image_at(0)
    assert isinstance(img, PILImage.Image)


def test_loader_get_image_out_of_bounds(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    with pytest.raises(IndexError):
        loader.image_at(999)


def test_loader_next_iterator(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    iterated = []
    for img_path in loader:
        iterated.append(img_path)
    assert len(iterated) == 3
    assert all(isinstance(p, ImagePath) for p in iterated)


def test_loader_reset(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    first_batch = list(loader)
    loader.reset()
    second_batch = list(loader)
    assert first_batch == second_batch


def test_loader_shuffle(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir), shuffle=True)
    paths = loader.all_image_paths()
    assert len(paths) == 3
