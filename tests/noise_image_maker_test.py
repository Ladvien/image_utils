import pytest
from pathlib import Path
from PIL import Image as PILImage
from image_utils.image_noiser import ImageNoiser
from image_utils.image_path import ImagePath
from image_utils.noising_operation import NosingOperation
from image_utils.noisy_image_maker import NoisyImageMaker


@pytest.fixture
def test_image_path(tmp_path):
    img_path = tmp_path / "image.jpg"
    img = PILImage.new("RGB", (32, 32), color="gray")
    img.save(img_path)
    return img_path


@pytest.fixture
def output_image_path(tmp_path):
    return tmp_path / "out.jpg"


@pytest.fixture
def image_path_obj(test_image_path):
    return ImagePath(str(test_image_path))


@pytest.fixture
def output_path_obj(output_image_path):
    return ImagePath(str(output_image_path))


@pytest.fixture
def noising_ops():
    return [
        NosingOperation(ImageNoiser.add_jpeg_compression, severity=0.5, order=0),
        NosingOperation(ImageNoiser.add_gaussian_noise, severity=0.3, order=1),
    ]


def test_noisy_image_maker_init(image_path_obj, output_path_obj, noising_ops):
    maker = NoisyImageMaker(
        image_path=image_path_obj,
        output_path=output_path_obj,
        noise_operations=noising_ops,
    )
    assert isinstance(maker, NoisyImageMaker)
    assert maker.validate()


def test_from_str_valid(test_image_path, output_image_path):
    maker = NoisyImageMaker.from_str(
        image_path=str(test_image_path),
        output_path=str(output_image_path),
        thresholds=[0.5, 0.2],
        noise_fns=["add_jpeg_compression", "add_gaussian_noise"],
    )
    assert isinstance(maker, NoisyImageMaker)
    assert len(maker.noise_operations) == 2


def test_from_str_mismatch_lengths_raises(test_image_path, output_image_path):
    with pytest.raises(ValueError):
        NoisyImageMaker.from_str(
            image_path=str(test_image_path),
            output_path=str(output_image_path),
            thresholds=[0.5],
            noise_fns=["add_jpeg_compression", "add_gaussian_noise"],
        )


def test_update_severity(image_path_obj, output_path_obj, noising_ops):
    maker = NoisyImageMaker(
        image_path=image_path_obj,
        output_path=output_path_obj,
        noise_operations=noising_ops,
    )
    maker.update_severity("add_gaussian_noise", 0.9)
    assert any(
        op.severity == 0.9
        for op in maker.noise_operations
        if op.name == "add_gaussian_noise"
    )


def test_update_severity_not_found_raises(image_path_obj, output_path_obj, noising_ops):
    maker = NoisyImageMaker(
        image_path=image_path_obj,
        output_path=output_path_obj,
        noise_operations=noising_ops,
    )
    with pytest.raises(ValueError):
        maker.update_severity("non_existent_noise", 0.5)


def test_set_fn_threshold(image_path_obj, output_path_obj, noising_ops):
    maker = NoisyImageMaker(
        image_path=image_path_obj,
        output_path=output_path_obj,
        noise_operations=noising_ops,
    )
    fn = ImageNoiser.add_jpeg_compression
    maker.set_fn_threshold(fn, 0.8)
    assert any(op.severity == 0.8 for op in maker.noise_operations if op.fn == fn)


def test_set_fn_threshold_invalid(image_path_obj, output_path_obj, noising_ops):
    maker = NoisyImageMaker(
        image_path=image_path_obj,
        output_path=output_path_obj,
        noise_operations=noising_ops,
    )
    with pytest.raises(ValueError):
        maker.set_fn_threshold(lambda x, y: x, 0.5)


def test_noisy_image_returns_image(image_path_obj, output_path_obj, noising_ops):
    maker = NoisyImageMaker(
        image_path=image_path_obj,
        output_path=output_path_obj,
        noise_operations=noising_ops,
    )
    img = maker.noisy_image()
    assert isinstance(img, PILImage.Image)


def test_noisy_base64(image_path_obj, output_path_obj, noising_ops):
    maker = NoisyImageMaker(
        image_path=image_path_obj,
        output_path=output_path_obj,
        noise_operations=noising_ops,
    )
    encoded = maker.noisy_base64()
    assert isinstance(encoded, str)
    assert encoded.startswith("iVBOR") or encoded.startswith(
        "/9j/"
    )  # PNG or JPEG base64
