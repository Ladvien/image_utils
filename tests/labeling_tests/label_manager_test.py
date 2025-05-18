# tests/labeling_tests/label_manager_test.py

import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from image_utils.noising_operation import NosingOperation
from image_utils.noisy_image_maker import NoisyImageMaker
from labeling.label_manager import LabelWriter, LabelManager

# -- Fixtures from your suite: tmp_image_dir, tmp_output_dir, etc.


@pytest.fixture(autouse=True)
def stub_nosing_operation(monkeypatch):
    """
    Replace NosingOperation.from_str with a factory that
    returns a simple object with the right attributes.
    """

    def fake_from_str(name, severity, order):
        return SimpleNamespace(name=name, severity=severity, order=order)

    monkeypatch.setattr(NosingOperation, "from_str", fake_from_str)
    yield


# Prevent randomness in LabelManager.new_noisy_image_maker()
@pytest.fixture(autouse=True)
def no_shuffle(monkeypatch):
    import labeling.label_manager as lm

    monkeypatch.setattr(lm, "shuffle", lambda x: None)
    yield


# Dummy config implementing the LabelManagerConfig interface
class DummyConfig:
    def __init__(
        self,
        images_dir,
        label_csv_path,
        output_dir,
        samples_per_image: int = 1,
        image_samples: int | None = None,
        shuffle_images: bool = False,
        overwrite_label_csv: bool = True,
    ):
        self.images_dir = images_dir
        self.label_csv_path = label_csv_path
        self.output_dir = output_dir
        self.samples_per_image = samples_per_image
        self.image_samples = image_samples
        self.shuffle_images = shuffle_images
        self.overwrite_label_csv = overwrite_label_csv

    def noise_fns_and_defaults(self):
        # two dummy noise fns
        return [("noise1", 0.3), ("noise2", 0.6)]

    def noise_function_names(self):
        return [name for name, _ in self.noise_fns_and_defaults()]


class DummyNoisy:
    """Minimal stand-in for NoisyImageMaker for LabelWriter tests."""

    def __init__(self, orig_path: str, ops):
        class P:
            def __init__(self, p):
                self.path = p

        self.image_path = P(orig_path)
        self.noise_operations = ops


def test_label_writer_creates_empty_csv(tmp_output_dir):
    csv_path = tmp_output_dir / "labels.csv"
    writer = LabelWriter(str(csv_path), max_operations=2)
    assert csv_path.exists()
    expected_cols = ["original_image_path", "label"] + [
        f"fn_{i}_{suf}" for i in (1, 2) for suf in ("name", "threshold", "order")
    ]
    assert writer.columns == expected_cols
    assert list(writer.df.columns) == expected_cols
    assert writer.num_labeled() == 0
    assert writer.get_labels() == []


def test_label_writer_record_and_persistence(tmp_output_dir):
    csv_path = tmp_output_dir / "labels.csv"
    writer = LabelWriter(str(csv_path), max_operations=1, overwrite=True)

    # create dummy operation via our stubbed from_str
    op = NosingOperation.from_str("noiseA", 0.5, 1)
    dummy = DummyNoisy("img/orig.jpg", [op])

    writer.record(dummy, "cat")
    # in-memory
    assert writer.num_labeled() == 1
    assert writer.get_labels() == ["img/orig.jpg"]

    # on-disk
    df = pd.read_csv(csv_path)
    assert df.shape == (1, len(writer.columns))
    row = df.iloc[0]
    assert row["original_image_path"] == "img/orig.jpg"
    assert row["label"] == "cat"
    assert row["fn_1_name"] == "noiseA"
    assert pytest.approx(row["fn_1_threshold"], 0.001) == 0.5
    assert int(row["fn_1_order"]) == 1


def test_label_manager_counts_and_progress(tmp_image_dir, tmp_output_dir):
    label_csv = tmp_output_dir / "labels.csv"
    out_dir = tmp_output_dir / "out"
    cfg = DummyConfig(
        images_dir=tmp_image_dir,
        label_csv_path=label_csv,
        output_dir=out_dir,
        image_samples=None,
    )
    manager = LabelManager(cfg)

    total_imgs = len(list(tmp_image_dir.rglob("*.jpg")))
    assert manager.total() == total_imgs
    assert manager.labeled_count() == 0
    assert manager.unlabeled_count() == total_imgs
    assert manager.percentage_complete() == 0.0

    # label one
    maker = manager.new_noisy_image_maker()
    assert isinstance(maker, NoisyImageMaker)
    manager.label_writer.record(maker, "dog")

    assert manager.labeled_count() == 1
    assert manager.unlabeled_count() == total_imgs - 1
    assert pytest.approx(manager.percentage_complete(), rel=1e-2) == 1 / total_imgs


def test_new_noisy_image_maker_exhaustion(tmp_image_dir, tmp_output_dir):
    label_csv = tmp_output_dir / "labels.csv"
    out_dir = tmp_output_dir / "out"
    cfg = DummyConfig(tmp_image_dir, label_csv, out_dir)
    mgr = LabelManager(cfg)

    seen = set()
    while True:
        nm = mgr.new_noisy_image_maker()
        if nm is None:
            break
        img = str(nm.image_path.path)
        assert img not in seen
        seen.add(img)
        mgr.label_writer.record(nm, "x")

    assert mgr.new_noisy_image_maker() is None
    assert mgr.labeled_count() == mgr.total()


def test_retrieve_records_roundtrip(tmp_image_dir, tmp_output_dir):
    label_csv = tmp_output_dir / "labels.csv"
    out_dir = tmp_output_dir / "out"
    cfg = DummyConfig(tmp_image_dir, label_csv, out_dir)
    mgr = LabelManager(cfg)

    # label two images
    for _ in range(2):
        nm = mgr.new_noisy_image_maker()
        mgr.label_writer.record(nm, "lbl")

    recs = mgr.retrieve_records()
    assert len(recs) == mgr.labeled_count()
    assert all(isinstance(r, NoisyImageMaker) for r in recs)
    original_paths = {str(r.image_path.path) for r in recs}
    assert original_paths == set(mgr.label_writer.get_labels())


def test_delete_last_label(tmp_image_dir, tmp_output_dir):
    label_csv = tmp_output_dir / "labels.csv"
    out_dir = tmp_output_dir / "out"
    cfg = DummyConfig(tmp_image_dir, label_csv, out_dir, samples_per_image=1)
    mgr = LabelManager(cfg)

    # nothing to delete
    assert mgr.delete_last_label() is False

    # label one and delete it
    nm = mgr.new_noisy_image_maker()
    mgr.label_writer.record(nm, "a")
    assert mgr.labeled_count() == 1
    assert mgr.delete_last_label() is True
    assert mgr.labeled_count() == 0
