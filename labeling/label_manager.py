from random import shuffle
from typing import Callable
from pathlib import Path
import pandas as pd
from rich import print

from image_utils.noising_operation import NosingOperation
from image_utils.noisy_image_maker import NoisyImageMaker
from image_utils.image_loader import ImageLoader
from image_utils.image_path import ImagePath
from labeling.label_manager_config import LabelManagerConfig


class LabelWriter:
    def __init__(
        self,
        path: str,
        image_path_column_name: str = "original_image_path",
        overwrite: bool = False,
        max_operations: int = 10,
    ):
        self.path = Path(path)
        self.max_operations = max_operations
        self.image_path_column_name = image_path_column_name

        self.columns = ["original_image_path", "label"] + [
            f"fn_{i}_{suffix}"
            for i in range(1, max_operations + 1)
            for suffix in ("name", "threshold", "order")
        ]

        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() or overwrite:
            self.df = pd.DataFrame(columns=self.columns)
            self.df.to_csv(self.path, index=False)
        else:
            print(f"Loading existing label CSV: {self.path}")
            self.df = pd.read_csv(self.path)

        assert list(self.df.columns) == self.columns

    def record(self, noisy_image_maker: NoisyImageMaker, label: str) -> None:
        row = {col: None for col in self.columns}
        row["original_image_path"] = noisy_image_maker.image_path.path
        row["label"] = label

        for i, op in enumerate(noisy_image_maker.noise_operations, start=1):
            row[f"fn_{i}_name"] = op.name
            row[f"fn_{i}_threshold"] = op.severity
            row[f"fn_{i}_order"] = op.order if hasattr(op, "order") else i

        print(f"Recording label: {row}")
        print(self.path)
        self.df.loc[len(self.df)] = row
        self.df.to_csv(self.path, index=False)

    def get_labels(self) -> list[str]:
        return self.df[self.image_path_column_name].tolist()

    def num_labeled(self) -> int:
        return len(self.df)


class LabelManager:
    def __init__(self, config: LabelManagerConfig):
        self.config = config
        self.image_loader = ImageLoader(
            config.images_dir, shuffle=config.shuffle_images
        )
        self.label_writer = LabelWriter(
            config.label_csv_path, overwrite=config.overwrite_label_csv
        )
        self.total_samples = config.image_samples or len(self.image_loader)

    def new_noisy_image_maker(self) -> NoisyImageMaker | None:
        try:
            while True:
                image_path = next(self.image_loader)
                if str(image_path.path) not in self.label_writer.get_labels():
                    noise_fn_defaults = self.config.noise_fns_and_defaults()
                    order = list(range(len(noise_fn_defaults)))
                    shuffle(order)
                    noise_ops = [
                        NosingOperation.from_str(
                            noise_fn_defaults[i][0], noise_fn_defaults[i][1], i
                        )
                        for i in order
                    ]
                    return NoisyImageMaker(
                        image_path, self.config.output_dir, noise_ops
                    )
        except StopIteration:
            return None

    def retrieve_records(self) -> list[NoisyImageMaker]:
        records = []
        for _, row in self.label_writer.df.iterrows():
            ops = []
            for i in range(1, self.label_writer.max_operations + 1):
                name = row.get(f"fn_{i}_name")
                threshold = row.get(f"fn_{i}_threshold")
                order = row.get(f"fn_{i}_order")

                if pd.notna(name) and pd.notna(threshold):
                    if name in self.config.noise_function_names():
                        ops.append(
                            NosingOperation.from_str(name, float(threshold), int(order))
                        )
            records.append(
                NoisyImageMaker(
                    ImagePath(row["original_image_path"]), self.config.output_dir, ops
                )
            )
        return records

    def unlabeled_count(self) -> int:
        return self.total_samples - self.labeled_count()

    def labeled_count(self) -> int:
        return len(self.label_writer.df)

    def percentage_complete(self) -> float:
        return (
            0.0
            if self.total_samples == 0
            else self.labeled_count() / self.total_samples
        )

    def total(self) -> int:
        return self.total_samples

    def delete_last_label(self) -> bool:
        df = self.label_writer.df
        if df.empty:
            print("No images to remove.")
            return False
        to_remove = min(self.config.samples_per_image, len(df))
        self.label_writer.df = df.drop(df.index[-to_remove:])
        self.label_writer.df.to_csv(self.label_writer.path, index=False)
        print(f"Removed last {to_remove} labeled images.")
        return True
