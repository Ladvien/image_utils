import pytest
from pathlib import Path
from PIL import Image as PILImage
from image_utils.image_path import ImagePath


@pytest.fixture
def valid_image(tmp_path):
    path = tmp_path / "valid.jpg"
    img = PILImage.new("RGB", (10, 10), color=(255, 0, 0))
    img.save(path)
    return path


@pytest.fixture
def non_image_file(tmp_path):
    path = tmp_path / "not_image.txt"
    path.write_text("This is not an image.")
    return path


def test_from_path_valid(valid_image):
    image_path = ImagePath.from_path(str(valid_image))
    assert image_path.path == str(valid_image)
    assert image_path.name == valid_image.name


def test_from_path_directory(tmp_path):
    with pytest.raises(IsADirectoryError):
        ImagePath.from_path(str(tmp_path))


def test_from_path_invalid(non_image_file):
    with pytest.raises(FileNotFoundError):
        ImagePath.from_path(str(non_image_file))


def test_load(valid_image):
    image_path = ImagePath.from_path(str(valid_image))
    image = image_path.load()
    assert isinstance(image, PILImage.Image)
    assert image.size == (10, 10)


def test_load_as_base64(valid_image):
    image_path = ImagePath.from_path(str(valid_image))
    encoded = image_path.load_as_base64()
    assert isinstance(encoded, str)
    assert len(encoded) > 0


def test_save(valid_image, tmp_path):
    image_path = ImagePath.from_path(str(valid_image))
    output_path = tmp_path / "nested/dir/saved.jpg"
    image_path.save(str(output_path))
    assert output_path.exists()
    loaded = PILImage.open(output_path)
    assert loaded.size == (10, 10)


def test_is_valid_image_true(valid_image):
    image_path = ImagePath.from_path(str(valid_image))
    assert image_path.is_valid_image() is True


def test_post_init_sets_name(valid_image):
    p = ImagePath(path=str(valid_image), name=None)
    assert p.name == valid_image.name
