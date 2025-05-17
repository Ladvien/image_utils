import sys
import os
import pytest
from PIL import Image as PILImage, UnidentifiedImageError, ImageFile
from io import BytesIO

# ensure project src is on PYTHONPATH
sys.path.insert(0, os.getcwd())

from image_utils.image_checker import ImageChecker

# allow loading truncated images in tests
ImageFile.LOAD_TRUNCATED_IMAGES = True


@pytest.fixture
def valid_image_path(tmp_path):
    p = tmp_path / "valid.jpg"
    img = PILImage.new("RGB", (10, 10), color="blue")
    img.save(p)
    return str(p)


@pytest.fixture
def corrupted_image_path(tmp_path):
    p = tmp_path / "corrupt.jpg"
    p.write_bytes(b"this is not a valid image")
    return str(p)


@pytest.fixture
def some_directory(tmp_path):
    d = tmp_path / "somedir"
    d.mkdir()
    return str(d)


def test_is_valid_image_with_valid(valid_image_path):
    assert ImageChecker.is_valid_image(valid_image_path) is True


def test_is_valid_image_with_nonexistent(tmp_path, capsys):
    fake = str(tmp_path / "does_not_exist.jpg")
    assert ImageChecker.is_valid_image(fake) is False
    captured = capsys.readouterr()
    assert "not found" in captured.out


def test_is_valid_image_with_directory(some_directory):
    assert ImageChecker.is_valid_image(some_directory) is False


def test_is_valid_image_with_corrupted(corrupted_image_path, capsys):
    assert ImageChecker.is_valid_image(corrupted_image_path) is False
    captured = capsys.readouterr()
    # should print the UnidentifiedImageError message
    assert "Invalid image" in captured.out


def test_can_be_reencoded_valid(valid_image_path):
    # valid image loads and verify passes
    assert ImageChecker.can_be_reencoded(valid_image_path) is True


def test_can_be_reencoded_corrupted(corrupted_image_path, capsys):
    # broken image should not verify
    assert ImageChecker.can_be_reencoded(corrupted_image_path) is False
    captured = capsys.readouterr()
    assert "cannot be re-encoded" in captured.out


def test_preview_image_valid_shows(monkeypatch, valid_image_path):
    called = {"yes": False}

    def fake_show(self):
        called["yes"] = True

    # stub out PIL.Image.Image.show
    monkeypatch.setattr(PILImage.Image, "show", fake_show)
    # should not error
    ImageChecker.preview_image(valid_image_path)
    assert called["yes"] is True


def test_preview_image_invalid(corrupted_image_path, capsys):
    # should catch error and print skip message
    ImageChecker.preview_image(corrupted_image_path)
    captured = capsys.readouterr()
    assert "cannot be previewed" in captured.out
