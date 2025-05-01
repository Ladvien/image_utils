import pytest
from pathlib import Path
from PIL import Image as PILImage
import shutil
import tempfile

from image_utils.image_loader import ImageLoader
from image_utils.image_noiser import ImageNoiser


@pytest.fixture
def tmp_image_dir():
    """Create a temporary directory with test images."""
    tmpdir = Path(tempfile.mkdtemp())
    for i in range(3):
        img = PILImage.new("RGB", (100, 100), color=(i * 50, i * 50, i * 50))
        img.save(tmpdir / f"test_image_{i}.jpg")
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture
def tmp_output_dir():
    """Create a temporary directory for noisy output."""
    tmpdir = Path(tempfile.mkdtemp())
    yield tmpdir
    shutil.rmtree(tmpdir)


def test_add_gaussian_noise(tmp_image_dir):
    image_path = list(tmp_image_dir.glob("*.jpg"))[0]
    image = PILImage.open(image_path)
    noisy = ImageNoiser.add_gaussian_noise(image, severity=0.2)
    assert isinstance(noisy, PILImage.Image)
    assert noisy.size == image.size


def test_add_jpeg_compression(tmp_image_dir):
    image_path = list(tmp_image_dir.glob("*.jpg"))[0]
    image = PILImage.open(image_path)
    compressed = ImageNoiser.add_jpeg_compression(image, severity=0.5)
    assert isinstance(compressed, PILImage.Image)
    assert compressed.size == image.size


def test_with_noise_combination(tmp_image_dir):
    image_path = list(tmp_image_dir.glob("*.jpg"))[0]
    image = PILImage.open(image_path)
    noisy = ImageNoiser.with_noise(image, severity=0.3)
    assert isinstance(noisy, PILImage.Image)
    assert noisy.size == image.size


def test_noise_images_saves_files(tmp_image_dir, tmp_output_dir):
    loader = ImageLoader(str(tmp_image_dir))
    # Using the noiser's own functions
    noise_functions = [
        lambda img, severity: ImageNoiser.add_gaussian_noise(img, severity),
        lambda img, severity: ImageNoiser.add_jpeg_compression(img, severity),
    ]

    ImageNoiser.noise_images(
        image_loader=loader,
        output_folder=str(tmp_output_dir),
        severity_range=(0.1, 0.5),
        noise_functions=noise_functions,
        samples=2,  # Don't process all
    )

    saved_files = list(tmp_output_dir.glob("*.jpg"))
    assert len(saved_files) == 2
    for file in saved_files:
        img = PILImage.open(file)
        assert isinstance(img, PILImage.Image)
