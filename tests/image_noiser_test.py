import pytest
import numpy as np
from PIL import Image as PILImage

from image_utils.image_noiser import ImageNoiser


@pytest.fixture
def sample_image():
    return PILImage.new("RGB", (32, 32), color=(128, 128, 128))


@pytest.mark.parametrize("severity", [0.0, 0.5, 1.0])
def test_add_gaussian_noise(sample_image, severity):
    result = ImageNoiser.add_gaussian_noise(sample_image, severity)
    assert isinstance(result, PILImage.Image)
    assert result.size == sample_image.size


@pytest.mark.parametrize("severity", [0.0, 0.5, 1.0])
def test_add_gaussian_blur(sample_image, severity):
    result = ImageNoiser.add_gaussian_blur(sample_image, severity)
    assert isinstance(result, PILImage.Image)
    assert result.size == sample_image.size


@pytest.mark.parametrize("severity", [0.0, 0.5, 1.0])
def test_add_jpeg_compression(sample_image, severity):
    result = ImageNoiser.add_jpeg_compression(sample_image, severity)
    assert isinstance(result, PILImage.Image)
    assert result.size == sample_image.size


@pytest.mark.parametrize("severity", [0.0, 0.5, 1.0])
def test_add_brightness(sample_image, severity):
    result = ImageNoiser.add_brightness(sample_image, severity)
    assert isinstance(result, PILImage.Image)


@pytest.mark.parametrize("severity", [0.0, 0.5, 1.0])
def test_add_high_contrast(sample_image, severity):
    result = ImageNoiser.add_high_contrast(sample_image, severity)
    assert isinstance(result, PILImage.Image)


@pytest.mark.parametrize("severity", [0.0, 0.5, 1.0])
def test_add_low_contrast(sample_image, severity):
    result = ImageNoiser.add_low_contrast(sample_image, severity)
    assert isinstance(result, PILImage.Image)


@pytest.mark.parametrize("severity", [0.0, 0.5, 1.0])
def test_add_saturation(sample_image, severity):
    result = ImageNoiser.add_saturation(sample_image, severity)
    assert isinstance(result, PILImage.Image)


@pytest.mark.parametrize("severity", [0.0, 0.5, 1.0])
def test_add_pixelate(sample_image, severity):
    result = ImageNoiser.add_pixelate(sample_image, severity)
    assert isinstance(result, PILImage.Image)
    assert result.size == sample_image.size


@pytest.mark.parametrize("severity", [0.0, 0.5, 1.0])
def test_with_noise(sample_image, severity):
    result = ImageNoiser.with_noise(sample_image, severity)
    assert isinstance(result, PILImage.Image)
    assert result.size == sample_image.size


def test_add_all_noises_typical(sample_image):
    img, weights = ImageNoiser.add_all_noises(
        sample_image, severity=1.5, num_noise_fns=3
    )
    assert isinstance(img, PILImage.Image)
    assert isinstance(weights, dict)
    assert img.size == sample_image.size
    assert pytest.approx(sum(weights.values()), abs=1e-5) == 1.5


def test_add_all_noises_zero(sample_image):
    img, weights = ImageNoiser.add_all_noises(
        sample_image, severity=0.0, num_noise_fns=3
    )
    assert isinstance(img, PILImage.Image)
    assert sum(weights.values()) == 0.0


def test_add_all_noises_exceeds_max(sample_image):
    with pytest.raises(ValueError, match="exceeds max total"):
        ImageNoiser.add_all_noises(sample_image, severity=10.0, num_noise_fns=2)


def test_add_all_noises_negative(sample_image):
    with pytest.raises(ValueError, match="Severity must be >= 0"):
        ImageNoiser.add_all_noises(sample_image, severity=-1.0, num_noise_fns=2)
