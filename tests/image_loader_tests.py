import pytest
from pathlib import Path
from PIL import Image as PILImage
import shutil
import tempfile

from image_utils.image_loader import ImageLoader
from image_utils.image_path import ImagePath


@pytest.fixture
def tmp_image_dir():
    """Create a temporary directory with test images."""
    tmpdir = Path(tempfile.mkdtemp())
    for i in range(5):
        img = PILImage.new("RGB", (100, 100), color=(i * 50, i * 50, i * 50))
        img.save(tmpdir / f"test_image_{i}.jpg")
    yield tmpdir
    shutil.rmtree(tmpdir)


def test_loader_finds_images(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    assert len(loader) == 5


def test_loader_iterates_images(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    paths = list(loader.iter_image_paths())
    assert all(isinstance(p, ImagePath) for p in paths)
    assert len(paths) == 5


def test_loader_iterates_loaded_images(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    images = list(loader.iter_images())
    assert all(isinstance(img, PILImage.Image) for img in images)
    assert len(images) == 5


def test_loader_indexing(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    img_path = loader[0]
    assert isinstance(img_path, ImagePath)


def test_loader_len(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    assert len(loader) == 5


def test_loader_get_all_image_paths(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    paths = loader.get_all_image_paths()
    assert len(paths) == 5
    assert all(isinstance(p, Path) or isinstance(p, str) for p in paths)


def test_loader_get_all_images(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    images = loader.get_all_images()
    assert len(images) == 5
    assert all(isinstance(img, PILImage.Image) for img in images)


def test_loader_get_image(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    img = loader.get_image(0)
    assert isinstance(img, PILImage.Image)


def test_loader_get_image_out_of_bounds(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    with pytest.raises(IndexError):
        loader.get_image(999)


def test_loader_next_iterator(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    iterated = []
    for img_path in loader:
        iterated.append(img_path)
    assert len(iterated) == 5
    assert all(isinstance(p, ImagePath) for p in iterated)


def test_loader_reset(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    first_batch = list(loader.iter_image_paths())
    loader.reset()
    second_batch = list(loader.iter_image_paths())
    assert first_batch == second_batch


def test_loader_shuffle(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir), shuffle=True)
    paths = loader.get_all_image_paths()
    assert len(paths) == 5  # Still should have all images
