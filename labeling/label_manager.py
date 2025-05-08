from typing import Callable
from image_utils.noising_operation import NosingOperation
from image_utils.noisy_image_maker import NoisyImageMaker
from labeling.label_manager_config import LabelManagerConfig
import pandas as pd
from pathlib import Path
from rich import print

from image_utils.image_loader import ImageLoader
from image_utils.image_path import ImagePath


class LabelWriter:
    def __init__(
        self,
        path: str,
        columns: list[str] = None,
        image_path_column_name: str = "original_image_path",
        overwrite: bool = False,
        max_operations: int = 10,
    ):
        self.path = Path(path)
        self.max_operations = max_operations

        # --- Define columns ---
        columns = ["original_image_path", "label"]  # Add label column first
        for i in range(1, self.max_operations + 1):
            columns.append(f"fn_{i}")
            columns.append(f"fn_{i}_threshold")

        self.columns = columns
        self.image_path_column_name = image_path_column_name

        self.path.parent.mkdir(parents=True, exist_ok=True)

        # --- Initialize CSV ---
        if not self.path.exists() or overwrite:
            self.df = pd.DataFrame(columns=self.columns)
            self.df.to_csv(self.path, index=False)
        else:
            print(f"Loading existing label CSV: {self.path}")
            self.df = pd.read_csv(self.path)

        # Make sure the loaded CSV has the right columns
        assert list(self.df.columns) == self.columns

    def record(self, noisy_image_maker: NoisyImageMaker, label: str) -> None:
        # --- Prepare a new row ---
        new_row = {col: None for col in self.columns}
        new_row["original_image_path"] = noisy_image_maker.image_path.path
        new_row["label"] = label

        for idx, op in enumerate(noisy_image_maker.noise_operations):
            fn_col = f"fn_{idx + 1}"
            threshold_col = f"fn_{idx + 1}_threshold"
            new_row[fn_col] = op.name
            new_row[threshold_col] = op.severity or 0

        # --- Append row ---
        self.df.loc[len(self.df)] = new_row
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
            config.label_csv_path, config.overwrite_label_csv
        )
        self.total_samples = config.image_samples or len(self.image_loader)

    def new_noisy_image_maker(self) -> NoisyImageMaker | None:
        try:
            while True:
                image_path = next(self.image_loader)
                if str(image_path.path) not in self.label_writer.get_labels():
                    noise_operations = [
                        NosingOperation.from_str(fn_name, default_threshold)
                        for fn_name, default_threshold in self.config.noise_fns_and_defaults()
                    ]
                    maker = NoisyImageMaker(
                        image_path,
                        self.config.output_dir,
                        noise_operations,
                    )
                    return maker
        except StopIteration:
            return None

    def unlabeled_count(self) -> int:
        return self.total_samples - self.labeled_count()

    def labeled_count(self) -> int:
        return len(self.label_writer.df)

    def percentage_complete(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.labeled_count() / self.total_samples

    def total(self) -> int:
        return self.total_samples

    def retrieve_records(self) -> list[NoisyImageMaker]:
        entries = []
        for _, row in self.label_writer.df.iterrows():
            noise_operations = []
            for i in range(1, self.label_writer.max_operations + 1):
                fn_name = row.get(f"fn_{i}")
                threshold = row.get(f"fn_{i}_threshold")

                if pd.notna(fn_name) and pd.notna(threshold):
                    if fn_name in self.config.noise_function_names():
                        noise_operations.append(
                            NosingOperation.from_str(fn_name, float(threshold))
                        )
                    else:
                        print(
                            f"Warning: function '{fn_name}' not found in noise_fn_names."
                        )

            entries.append(
                NoisyImageMaker(
                    image_path=ImagePath(row["original_image_path"]),
                    output_path=self.config.output_dir,
                    noise_operations=noise_operations,
                )
            )

        return entries

    def set_noise_fn(self, fn: Callable):
        self.noise_fn_names = fn

    def delete_last_label(self) -> bool:
        """
        Removes the last num_images labeled images from the label writer.
        Returns True if images were removed, False otherwise.
        """
        df = self.label_writer.df

        if df.empty:
            print("No images to remove.")
            return False

        safe_num_to_remove = min(self.config.samples_per_image, len(df))
        rows_to_delete = df.index[-safe_num_to_remove:]
        df = df.drop(rows_to_delete)
        df.to_csv(self.label_writer.path, index=False)
        self.label_writer.df = df
        print(f"Removed last {safe_num_to_remove} labeled images.")
        return True
