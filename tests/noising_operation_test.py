import pytest
from PIL import Image as PILImage
from image_utils.noising_operation import NosingOperation
from image_utils.image_noiser import ImageNoiser


@pytest.fixture
def dummy_image():
    return PILImage.new("RGB", (32, 32), color=(128, 128, 128))


def test_init_and_call(dummy_image):
    op = NosingOperation(ImageNoiser.add_gaussian_blur, severity=0.5, order=1)
    result = op(dummy_image, op.severity)
    assert isinstance(result, PILImage.Image)
    assert result.size == dummy_image.size
    assert op.name == "add_gaussian_blur"
    assert op.order == 1
    assert op.severity == 0.5


def test_from_str_valid(dummy_image):
    op = NosingOperation.from_str("add_gaussian_noise", 0.3, 2)
    assert isinstance(op, NosingOperation)
    assert op.name == "add_gaussian_noise"
    result = op(dummy_image, op.severity)
    assert isinstance(result, PILImage.Image)


def test_from_str_invalid():
    with pytest.raises(AttributeError):
        NosingOperation.from_str("nonexistent_noise_fn", 0.5, 1)


def test_to_dict():
    op = NosingOperation(ImageNoiser.add_jpeg_compression, severity=0.2, order=3)
    d = op.to_dict()
    assert isinstance(d, dict)
    assert d["name"] == "add_jpeg_compression"
    assert d["severity"] == 0.2
    assert d["order"] == 3


def test_repr():
    op = NosingOperation(ImageNoiser.add_gaussian_blur, severity=0.7, order=5)
    text = repr(op)
    assert "NosingOperation" in text
    assert "add_gaussian_blur" in text
    assert "severity=0.7" in text
    assert "order=5" in text
