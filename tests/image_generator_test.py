# tests/test_pipeline.py
import sys
import os
import pytest
import pandas as pd
from pathlib import Path
from PIL import Image as PILImage

# Make sure your project root is on the PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Adjust this import to where your pipeline lives:
from image_utils.image_generator import Config, ImageTransformationPipeline


def test_config_paths(tmp_path):
    out = tmp_path / "out"
    cfg = Config(
        source_dir=tmp_path / "src",
        output_dir=out,
        image_transform_fn=lambda img: (img, {}),
    )
    assert cfg.train_dir() == out / "train"
    assert cfg.test_dir() == out / "test"
    assert cfg.output_csv_path() == out / "noisy_labels.csv"


def test_pipeline_end_to_end(tmp_path, sample_src, dummy_noise_fn):
    src = sample_src
    out = tmp_path / "out"
    # 50% test split so 2 images => 1 train, 1 test
    cfg = Config(
        source_dir=src,
        output_dir=out,
        image_transform_fn=dummy_noise_fn,
        test_size=0.5,
        shuffle=False,
    )
    pipeline = ImageTransformationPipeline(cfg)
    pipeline.run()

    # 1) Check that train/test dirs each have exactly one output .jpg
    train_files = list((out / "train").glob("*.jpg"))
    test_files = list((out / "test").glob("*.jpg"))
    assert len(train_files) == 1, "Expected 1 train image"
    assert len(test_files) == 1, "Expected 1 test image"

    # 2) CSV exists with 2 rows
    csv_path = out / "noisy_labels.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) == 2

    # 3) Columns include original, noisy, split, and dummy_noise
    for col in ("original_image_path", "noisy_image_path", "split", "dummy_noise"):
        assert col in df.columns

    # 4) The original paths in CSV match exactly the two files we created
    expected = {str(p) for p in (src / "a" / "a.jpg",).__iter__()} | {
        str(p) for p in (src / "b" / "b.jpg",).__iter__()
    }
    # Because we know the two exact paths:
    expected = {str(src / "a" / "a.jpg"), str(src / "b" / "b.jpg")}
    assert set(df["original_image_path"]) == expected
