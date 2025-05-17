import shutil
import tempfile
from pathlib import Path
import pytest
from PIL import Image as PILImage

TEMP_ROOT = Path("tests") / "temp_data"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def tmp_image_dir():
    """Create a temp directory with test images."""
    tmpdir = TEMP_ROOT / "images"
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    tmpdir.mkdir(parents=True)

    for i in range(3):
        img = PILImage.new("RGB", (100, 100), color=(i * 50, i * 50, i * 50))
        img.save(tmpdir / f"test_image_{i}.jpg")

    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture
def tmp_output_dir():
    """Create a temp directory for noisy output."""
    tmpdir = TEMP_ROOT / "output"
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    tmpdir.mkdir(parents=True)

    yield tmpdir


@pytest.fixture
def tmp_non_image_file():
    """Create a temporary non-image file."""
    tmpdir = Path(tempfile.mkdtemp())
    file_path = tmpdir / "not_an_image.txt"
    file_path.write_text("This is not an image.")
    yield file_path
    shutil.rmtree(tmpdir)


@pytest.fixture
def tmp_image_file():
    """Create a temporary valid image file."""
    tmpdir = Path(tempfile.mkdtemp())
    img_path = tmpdir / "test_image.jpg"
    img = PILImage.new("RGB", (64, 64), color=(100, 150, 200))
    img.save(img_path)
    yield img_path
    shutil.rmtree(tmpdir)


@pytest.fixture
def sample_src(tmp_path):
    """
    Create a small nested directory of two dummy JPG images:
      src/a/a.jpg
      src/b/b.jpg
    """
    base = tmp_path / "src"
    for folder, color in [("a", (255, 0, 0)), ("b", (0, 255, 0))]:
        d = base / folder
        d.mkdir(parents=True)
        img = PILImage.new("RGB", (8, 8), color=color)
        img.save(d / f"{folder}.jpg")
    return base


@pytest.fixture
def dummy_noise_fn():
    """
    A stand-in transform that simply flips the image and returns a fixed
    weight dict, matching the pipeline's expectation of a (img, weights) tuple.
    """

    def fn(image):
        flipped = image.transpose(PILImage.FLIP_LEFT_RIGHT)
        return flipped, {"dummy_noise": 1.0}

    return fn


@pytest.fixture
def tmp_single_image(tmp_path):
    """A single valid PNG for re-encode tests."""
    d = tmp_path / "single"
    d.mkdir()
    p = d / "one.png"
    PILImage.new("RGB", (5, 5), color=(1, 2, 3)).save(p)
    return p
