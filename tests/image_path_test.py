import pytest
from pathlib import Path
from PIL import Image as PILImage
import shutil
import tempfile
import base64

from image_utils.image_path import ImagePath


def test_imagepath_creation(tmp_image_file):
    img_path = ImagePath(path=str(tmp_image_file))
    assert img_path.path == str(tmp_image_file)
    assert img_path.name == tmp_image_file.name


def test_imagepath_from_path(tmp_image_file):
    img_path = ImagePath.from_path(str(tmp_image_file))
    assert isinstance(img_path, ImagePath)
    assert img_path.name == tmp_image_file.name


def test_load_image(tmp_image_file):
    img_path = ImagePath(path=str(tmp_image_file))
    image = img_path.load()
    assert isinstance(image, PILImage.Image)
    assert image.size == (64, 64)


def test_load_as_base64(tmp_image_file):
    img_path = ImagePath(path=str(tmp_image_file))
    img_b64 = img_path.load_as_base64()
    assert isinstance(img_b64, str)
    # Check if it decodes successfully
    decoded = base64.b64decode(img_b64)
    assert isinstance(decoded, bytes)


def test_save(tmp_image_file):
    img_path = ImagePath(path=str(tmp_image_file))
    tmpdir = Path(tempfile.mkdtemp())
    save_path = tmpdir / "saved_image.jpg"

    img_path.save(str(save_path))
    assert save_path.exists()

    # Confirm saved image is valid
    reloaded = PILImage.open(save_path)
    assert reloaded.size == (64, 64)

    shutil.rmtree(tmpdir)


def test_is_valid_image(tmp_image_file):
    img_path = ImagePath(path=str(tmp_image_file))
    assert img_path.is_valid_image()


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        ImagePath.from_path("non_existent_file.jpg")


def test_directory_path_raises(tmp_image_file):
    directory = tmp_image_file.parent
    with pytest.raises(IsADirectoryError):
        ImagePath.from_path(path=str(directory))
