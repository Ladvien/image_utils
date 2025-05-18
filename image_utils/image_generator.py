from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, List, Dict, Optional
from uuid import uuid4
from PIL import Image as PILImage
from image_utils.image_noiser import ImageNoiser
from image_utils.utils import coerce_to_paths
import pandas as pd
import os
from sklearn.model_selection import train_test_split

from image_utils.image_loader import ImageLoader  # Your lazy recursive loader


@dataclass
class Config:
    source_dir: List[Path] | Path
    output_dir: Path
    image_transform_fn: Callable[
        [PILImage.Image, Path], tuple[PILImage.Image, Dict[str, float]]
    ]
    test_size: float = 0.2
    output_csv_name: str = "noisy_labels.csv"
    train_subdir: str = "train"
    test_subdir: str = "test"
    image_extensions: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png"]
    )
    shuffle: bool = True

    def train_dir(self) -> Path:
        return self.output_dir / self.train_subdir

    def test_dir(self) -> Path:
        return self.output_dir / self.test_subdir

    def output_csv_path(self) -> Path:
        return self.output_dir / self.output_csv_name

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self.source_dir = coerce_to_paths(self.source_dir)
        if not self.output_dir.exists():
            raise ValueError(f"Output directory does not exist: {self.output_dir}")
        if not self.image_extensions:
            raise ValueError("No image extensions provided.")
        if self.test_size <= 0 or self.test_size >= 1:
            raise ValueError("test_size must be between 0 and 1.")
        if not callable(self.image_transform_fn):
            raise ValueError("image_transform_fn must be a callable.")
        if not isinstance(self.shuffle, bool):
            raise ValueError("shuffle must be a boolean.")


class ImageTransformationPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.image_loader = ImageLoader(
            str(config.source_dir),
            extensions=config.image_extensions,
            shuffle=config.shuffle,
        )
        print(f"✅ Loaded {len(self.image_loader)} images from {config.source_dir}")

        self.records: List[Dict] = []
        self._setup_output_dirs()

    def _setup_output_dirs(self):
        self.config.train_dir().mkdir(parents=True, exist_ok=True)
        self.config.test_dir().mkdir(parents=True, exist_ok=True)
        print(
            f"✅ Output dirs created:\n- {self.config.train_dir()}\n- {self.config.test_dir()}"
        )

    def run(self):
        if len(self.image_loader) < 2:
            raise ValueError("Not enough images to split into train and test sets.")

        image_paths = [p.path for p in self.image_loader]
        self.image_loader.reset()

        train_paths, test_paths = train_test_split(
            image_paths,
            test_size=self.config.test_size,
            random_state=42,
        )
        split_lookup = {str(p): "train" for p in train_paths} | {
            str(p): "test" for p in test_paths
        }

        for img_path_obj in self.image_loader:
            img_path = img_path_obj.path
            try:
                image = img_path_obj.load()
            except Exception as e:
                print(f"❌ Skipping {img_path} due to error: {e}")
                continue

            group = split_lookup.get(str(img_path), None)
            if group not in ("train", "test"):
                print(f"❓ Image not in split map: {img_path}")
                continue

            dest_dir = (
                self.config.train_dir() if group == "train" else self.config.test_dir()
            )
            output_filename = f"{uuid4()}_{Path(img_path).stem}_noisy.jpg"
            output_path = dest_dir / output_filename

            if output_path.exists():
                print(f"⚠️  Already exists: {output_path}")
                continue

            transformed_image, noise_weights = self.config.image_transform_fn(image)
            print(noise_weights)

            transformed_image.save(output_path, quality=100)
            print(f"✅ Saved: {group}/{output_filename}")

            record = {
                "original_image_path": str(img_path),
                "noisy_image_path": str(output_path),
                "split": group,
            } | {name: str(weight) for name, weight in noise_weights.items()}

            print(f"📝 Record: {record}")
            self.records.append(record)

        self._write_csv()

    def _write_csv(self):
        df = pd.DataFrame(self.records)
        df.to_csv(self.config.output_csv_path(), index=False)
        print(f"📝 Saved labels to {self.config.output_csv_path()}")


if __name__ == "__main__":

    maker = ImageNoiser()

    add_noises_partial = partial(
        maker.add_all_noises, severity_budget=1.2, num_noise_fns=3
    )

    config = Config(
        source_dir=Path("/mnt/storage/external_ssd/datasets/laion_aesthetics/50/"),
        output_dir=Path("/mnt/datadrive_m2/ml_training_data/aiqa"),
        # WILO: Make `add_all_noises` a callable that takes an image and a path
        image_transform_fn=add_noises_partial,
    )
    pipeline = ImageTransformationPipeline(config)
    pipeline.run()
