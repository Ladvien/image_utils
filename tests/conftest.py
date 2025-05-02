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
