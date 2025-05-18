import pytest
import numpy as np
from PIL import Image as PILImage

from image_utils.image_noiser import ImageNoiser


@pytest.fixture
def sample_image():
    # 32×32 mid‐gray
    return PILImage.new("RGB", (32, 32), color=(128, 128, 128))


def test_get_noise_functions_contains_expected():
    fns = ImageNoiser.get_noise_functions()
    names = {fn.__name__ for fn in fns}
    expected = {
        "add_gaussian_noise",
        "add_jpeg_compression",
        "add_gaussian_blur",
        "add_brightness",
        "add_high_contrast",
        "add_low_contrast",
        "add_saturation",
        "add_pixelate",
    }
    # you may have more, but at least these must be present
    assert expected.issubset(names)


@pytest.mark.parametrize("n", [1, 3, None])
def test_select_noise_functions_length(n):
    all_fns = ImageNoiser.get_noise_functions()
    sel = ImageNoiser._select_noise_functions(n)
    # if n=None, expect all; else minimum of n and len(all)
    expected_len = len(all_fns) if n is None else min(n, len(all_fns))
    assert len(sel) == expected_len
    assert all(callable(fn) for fn in sel)


@pytest.mark.parametrize(
    "n,budget",
    [
        (1, 0.0),
        (3, 1.5),
        (5, 2.3),
    ],
)
def test_compute_weights_properties(n, budget):
    weights = ImageNoiser._compute_weights(n, budget)
    assert len(weights) == n
    # sum matches budget
    assert pytest.approx(sum(weights), rel=1e-6) == budget
    # all non-negative
    assert all(w >= 0 for w in weights)


def test_blend_noise_two_funcs(sample_image):
    # use two simple funcs: noising + brightness
    funcs = [
        ImageNoiser.add_gaussian_noise,
        ImageNoiser.add_brightness,
    ]
    weights = [0.3, 0.7]
    blended, applied = ImageNoiser._blend_noise(sample_image, funcs, weights)
    assert isinstance(blended, PILImage.Image)
    assert blended.size == sample_image.size
    # mapping of names→weights matches
    assert applied == {"add_gaussian_noise": 0.3, "add_brightness": 0.7}


# individual noise ops
@pytest.mark.parametrize("fn", ImageNoiser.get_noise_functions())
@pytest.mark.parametrize("severity", [0.0, 0.5, 1.0])
def test_individual_ops_preserve_size(fn, sample_image, severity):
    out = fn(sample_image, severity)
    assert isinstance(out, PILImage.Image)
    assert out.size == sample_image.size


def test_add_jpeg_compression_mode_conversion(sample_image):
    # ensure non-RGB modes are handled
    gray = sample_image.convert("L")
    out = ImageNoiser.add_jpeg_compression(gray, 0.5)
    assert out.mode == "RGB"
    assert out.size == gray.size


def test_add_all_noises_typical(sample_image):
    img, weights = ImageNoiser.add_all_noises(
        sample_image, severity_budget=1.5, num_noise_fns=3
    )
    assert isinstance(img, PILImage.Image)
    assert img.size == sample_image.size
    assert isinstance(weights, dict)
    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.5


def test_add_all_noises_zero_budget(sample_image):
    img, weights = ImageNoiser.add_all_noises(sample_image, severity_budget=0.0)
    assert isinstance(img, PILImage.Image)
    assert sum(weights.values()) == 0.0


def test_add_all_noises_negative_budget(sample_image):
    with pytest.raises(ValueError, match="must be in"):
        ImageNoiser.add_all_noises(sample_image, severity_budget=-0.1)


def test_add_all_noises_exceeds_budget(sample_image):
    # if you request more budget than available funcs → error
    with pytest.raises(ValueError, match="must be in"):
        ImageNoiser.add_all_noises(sample_image, severity_budget=999.0, num_noise_fns=2)
