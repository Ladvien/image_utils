# tests/test_image_loader.py

import os
import pytest
import random
from pathlib import Path
from PIL import Image as PILImage
from image_utils.image_loader import ImageLoader
from image_utils.image_path import ImagePath


def create_image(path: Path, color=(100, 100, 100), ext=".jpg"):
    img = PILImage.new("RGB", (10, 10), color=color)
    img.save(path.with_suffix(ext))
    return path.with_suffix(ext)


def test_len_and_getitem(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    assert len(loader) == 3
    expected_names = sorted(p.name for p in tmp_image_dir.iterdir())
    assert loader[0].name == expected_names[0]


def test_iteration_and_reset(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    paths = list(loader)
    assert all(isinstance(p, ImagePath) for p in paths)
    assert len(paths) == 3
    loader.reset()
    assert len(loader) == 3


def test_all_images_and_paths(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    imgs = loader.all_images()
    paths = loader.all_image_paths()
    assert len(imgs) == len(paths) == 3
    assert all(isinstance(i, PILImage.Image) for i in imgs)
    assert all(isinstance(p, str) for p in paths)


def test_image_at_and_path_at(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    assert isinstance(loader.image_at(0), PILImage.Image)
    assert isinstance(loader.image_path_at(0), ImagePath)


def test_total_matches_len(tmp_image_dir):
    loader = ImageLoader(str(tmp_image_dir))
    assert loader.total() == len(loader)


def test_paths_with_image_extension(tmp_image_dir, tmp_non_image_file):
    mixed = list(tmp_image_dir.iterdir()) + [tmp_non_image_file]
    loader = ImageLoader(str(tmp_image_dir))
    filtered = loader.paths_with_image_extension(mixed)
    assert all(p.suffix.lower() == ".jpg" for p in filtered)
    with pytest.raises(Exception):
        loader.paths_with_image_extension([tmp_non_image_file])


def test_discover_image_paths(sample_src):
    loader = ImageLoader(str(sample_src))
    paths = loader.discover_image_paths(str(sample_src))
    assert all(p.suffix.lower() == ".jpg" for p in paths)
    assert len(paths) == 2


def test_shuffle_invoked(monkeypatch, tmp_image_dir):
    flag = {"called": False}

    def fake_shuffle(lst):
        flag["called"] = True

    monkeypatch.setattr(random, "shuffle", fake_shuffle)
    ImageLoader(str(tmp_image_dir), shuffle=True)
    assert flag["called"]


def test_shuffle_skipped(monkeypatch, tmp_image_dir):
    flag = {"called": False}

    def fake_shuffle(lst):
        flag["called"] = True

    monkeypatch.setattr(random, "shuffle", fake_shuffle)
    ImageLoader(str(tmp_image_dir), shuffle=False)
    assert not flag["called"]


def test_attempt_reencode_behavior(tmp_single_image, tmp_non_image_file, capsys):
    loader = ImageLoader(str(tmp_single_image.parent))

    assert loader.attempt_reencode(str(tmp_single_image)) is True
    assert "Re-encoded" in capsys.readouterr().out

    assert loader.attempt_reencode(str(tmp_non_image_file)) is False
    assert "Failed to re-encode" in capsys.readouterr().out


def test_empty_directory(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(Exception, match="No files found"):
        ImageLoader(str(empty))


def test_nested_empty_directory(tmp_path):
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    with pytest.raises(Exception, match="No files found"):
        ImageLoader(str(tmp_path))


def test_hidden_files_ignored(tmp_path):
    d = tmp_path / "img"
    d.mkdir()
    create_image(d / "real")
    (d / ".DS_Store").write_text("meta")
    (d / "._junk.jpg").write_text("meta")
    loader = ImageLoader(str(d))
    assert len(loader) == 1


def test_uppercase_extensions(tmp_path):
    d = tmp_path / "upper"
    d.mkdir()
    create_image(d / "CAPS", ext=".JPG")
    loader = ImageLoader(str(d))
    assert len(loader) == 1


def test_symlink_image(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    real = create_image(src / "real")
    (dst / "link.jpg").symlink_to(real)
    loader = ImageLoader(str(dst))
    assert len(loader) == 1


def test_unicode_filename(tmp_path):
    d = tmp_path / "unicode"
    d.mkdir()
    create_image(d / "🌟_图像")
    loader = ImageLoader(str(d))
    assert len(loader) == 1


def test_readonly_file(tmp_path):
    d = tmp_path / "ro"
    d.mkdir()
    img = create_image(d / "image")
    os.chmod(img, 0o444)
    loader = ImageLoader(str(d))
    assert len(loader) == 1
    os.chmod(img, 0o666)


def test_permission_denied(monkeypatch, tmp_path):
    d = tmp_path / "bad"
    d.mkdir()

    def deny(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "rglob", deny)
    with pytest.raises(PermissionError, match="denied"):
        ImageLoader(str(d))
