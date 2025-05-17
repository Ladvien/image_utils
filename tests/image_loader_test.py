# tests/test_image_loader.py

import pytest
import random
from pathlib import Path
from PIL import Image as PILImage

# Adjust these imports to match your project layout!
# For example, if your loader sits in image_utils/image_loader.py:
from image_utils.image_loader import ImageLoader
from image_utils.image_path import ImagePath


def test_len_and_getitem(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    assert len(loader) == 3
    # sorted by filename when shuffle=False
    names = sorted(p.name for p in tmp_image_dir.iterdir())
    assert loader[0].name == names[0]


def test_iteration_and_reset(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    first = next(loader)
    assert isinstance(first, ImagePath)
    assert len(loader) == 2
    loader.reset()
    assert len(loader) == 3
    # exhaust iterator
    it = iter(loader)
    for _ in range(3):
        next(it)
    with pytest.raises(StopIteration):
        next(it)


def test_all_images_and_all_image_paths(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    imgs = loader.all_images()
    paths = loader.all_image_paths()
    assert len(imgs) == 3
    assert all(isinstance(im, PILImage.Image) for im in imgs)
    assert len(paths) == 3
    assert all(isinstance(p, str) for p in paths)


def test_image_at_and_image_path_at(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    img = loader.image_at(1)
    ip = loader.image_path_at(1)
    assert isinstance(img, PILImage.Image)
    assert isinstance(ip, ImagePath)


def test_total_method(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    assert loader.total() == len(loader)


def test_paths_with_image_extension(tmp_image_dir, tmp_non_image_file):
    raw = list(tmp_image_dir.iterdir()) + [tmp_non_image_file]
    loader = ImageLoader(str(tmp_image_dir))
    filtered = loader.paths_with_image_extension(raw)
    # only .jpg should remain
    assert all(p.suffix.lower() == ".jpg" for p in filtered)
    # if none match, exception
    with pytest.raises(Exception):
        loader.paths_with_image_extension([tmp_non_image_file])


def test_discover_image_paths(sample_src):
    loader = ImageLoader(str(sample_src))
    found = loader.discover_image_paths(str(sample_src))
    assert set(p.suffix.lower() for p in found) == {".jpg"}
    assert len(found) == 2


def test_shuffle_image_paths_invoked(monkeypatch, tmp_image_dir):
    called = {"shuffled": False}

    def fake_shuffle(lst):
        called["shuffled"] = True

    monkeypatch.setattr(random, "shuffle", fake_shuffle)
    # shuffle=True should call it
    loader = ImageLoader(str(tmp_image_dir), shuffle=True)
    assert called["shuffled"]
    called["shuffled"] = False
    # shuffle=False should not
    loader2 = ImageLoader(str(tmp_image_dir), shuffle=False)
    assert not called["shuffled"]


def test_attempt_reencode(tmp_single_image, tmp_non_image_file, capsys):
    loader = ImageLoader(str(tmp_single_image.parent))
    # valid image -> True + prints "Re-encoded"
    ok = loader.attempt_reencode(str(tmp_single_image))
    out = capsys.readouterr().out
    assert ok is True
    assert "Re-encoded" in out
    # non-image -> False + prints "Failed to re-encode"
    ok2 = loader.attempt_reencode(str(tmp_non_image_file))
    out2 = capsys.readouterr().out
    assert ok2 is False
    assert "Failed to re-encode" in out2


import pytest
import os
from pathlib import Path
from PIL import Image as PILImage
from image_utils.image_loader import ImageLoader


def create_image(path: Path, color=(100, 100, 100), ext=".jpg"):
    img = PILImage.new("RGB", (10, 10), color=color)
    img.save(path.with_suffix(ext))
    return path.with_suffix(ext)


def test_empty_directory(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(Exception, match="No files found"):
        ImageLoader(str(empty_dir))


def test_nested_empty_directories(tmp_path):
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    with pytest.raises(Exception, match="No files found"):
        ImageLoader(str(tmp_path))


def test_ignore_ds_store_and_hidden_files(tmp_path):
    img_dir = tmp_path / "img"
    img_dir.mkdir()
    create_image(img_dir / "valid")
    (img_dir / ".DS_Store").write_text("metadata")
    (img_dir / "._junk.jpg").write_text("junk")
    loader = ImageLoader(str(img_dir))
    assert len(loader) == 1


def test_uppercase_extension(tmp_path):
    img_dir = tmp_path / "caps"
    img_dir.mkdir()
    img_path = img_dir / "IMAGE.JPG"
    create_image(img_path)
    loader = ImageLoader(str(img_dir))
    assert len(loader) == 1


def test_symlinked_image(tmp_path):
    src_dir = tmp_path / "source"
    dst_dir = tmp_path / "dest"
    src_dir.mkdir()
    dst_dir.mkdir()
    real_img = create_image(src_dir / "real")
    symlink_path = dst_dir / "linked.jpg"
    symlink_path.symlink_to(real_img)
    loader = ImageLoader(str(dst_dir))
    assert len(loader) == 1


def test_image_with_unicode_name(tmp_path):
    img_dir = tmp_path / "unicode"
    img_dir.mkdir()
    img_path = img_dir / "🌈_图像.jpg"
    create_image(img_path)
    loader = ImageLoader(str(img_dir))
    assert len(loader) == 1


def test_read_only_image(tmp_path):
    img_dir = tmp_path / "readonly"
    img_dir.mkdir()
    img_path = create_image(img_dir / "ro_img")
    os.chmod(img_path, 0o444)  # read-only
    loader = ImageLoader(str(img_dir))
    assert len(loader) == 1
    os.chmod(img_path, 0o666)  # restore permissions


def test_folder_permission_denied(monkeypatch, tmp_path):
    bad_dir = tmp_path / "restricted"
    bad_dir.mkdir()

    def fake_rglob(*args, **kwargs):
        raise PermissionError("Mocked permission denied")

    monkeypatch.setattr(Path, "rglob", fake_rglob)
    with pytest.raises(PermissionError, match="Mocked permission denied"):
        ImageLoader(str(bad_dir))
