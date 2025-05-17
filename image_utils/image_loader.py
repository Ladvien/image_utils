from glob import glob
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Optional
from PIL import Image as PILImage, ImageFile
import random
import os

from .image_path import ImagePath

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageLoader:
    def __init__(
        self,
        input_folder: str,
        extensions: Optional[List[str]] = None,
        shuffle: Optional[bool] = False,
    ):
        self.input_folder = input_folder
        self.shuffle = shuffle
        if extensions:
            self.extensions = [ext.lower() for ext in extensions]
        else:
            self.extensions = [".jpg", ".jpeg", ".png"]

        self._setup(input_folder)

    def __iter__(self):
        """Return self as an iterator."""
        return self

    def __next__(self) -> ImagePath:
        """Return the next ImagePath."""
        if self.image_paths:
            return self.image_paths.pop(0)
        else:
            raise StopIteration

    def __getitem__(self, index: int) -> ImagePath:
        """Allow index access to ImagePath objects."""
        return self.image_paths[index]

    def __len__(self) -> int:
        """Allow len(loader) to return number of images."""
        return len(self.image_paths)

    def image_paths(self) -> Iterable[ImagePath]:
        """Return a new iterator over image paths."""
        return (image_path for image_path in self.image_paths)

    def images(self) -> Iterable[PILImage.Image]:
        """Yield all loaded PIL Images."""
        return (img_path.load() for img_path in self.image_paths)

    def all_images(self) -> List[PILImage.Image]:
        """Return all images eagerly."""
        return [image_path.load() for image_path in self.image_paths]

    def all_image_paths(self) -> List[str]:
        """Return all file paths eagerly."""
        return [image_path.path for image_path in self.image_paths]

    def image_at(self, index: int) -> PILImage.Image:
        """Return a loaded image by index."""
        return self.image_path_at(index).load()

    def image_path_at(self, index: int) -> ImagePath:
        """Return an ImagePath by index."""
        return self[index]

    def total(self) -> int:
        """Return the total number of images."""
        return len(self)

    def reset(self):
        """Reset the iterator to the beginning."""
        self._setup(self.input_folder)

    def paths_with_image_extension(self, raw_image_paths: List[Path]) -> List[Path]:
        """Filter paths to include only those with valid image extensions."""
        image_paths = [
            path
            for path in raw_image_paths
            if path.suffix.lower() in self.extensions
            and not path.name.startswith("._")  # Skip macOS metadata files
        ]

        if not image_paths:
            raise Exception(
                f"No files found in '{self.input_folder}' with extensions {self.extensions}."
            )

        return image_paths

    def discover_image_paths(self, input_folder: str) -> List[Path]:
        """Discover all image paths in the input folder."""
        raw_image_paths = list(Path(input_folder).rglob("*", case_sensitive=False))
        if not raw_image_paths:
            raise Exception(f"No files found in '{self.input_folder}'.")

        image_paths = self.paths_with_image_extension(raw_image_paths)

        return image_paths

    def shuffle_image_paths(self, image_paths: List[Path]) -> None:
        """Shuffle the image paths in place."""
        if self.shuffle:
            random.shuffle(image_paths)
        else:
            image_paths.sort()

    def attempt_reencode(self, path: str) -> bool:
        """Attempt to re-encode an image file."""
        try:
            with PILImage.open(path) as img:
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=100)
                buffer.seek(0)
                new_image = PILImage.open(buffer)
                new_image.save(path, format="JPEG", quality=100)

                print(f"Re-encoded: {path}")
                return True
        except Exception as e:
            print(f"Failed to re-encode {path}: {e}")
            return False

    def _setup(self, input_folder: str):
        filtered_image_paths = self.discover_image_paths(input_folder)
        self.shuffle_image_paths(filtered_image_paths)
        potential_image_paths = [ImagePath(str(path)) for path in filtered_image_paths]

        self.image_paths = []
        for path in potential_image_paths:
            if path.is_valid_image():
                self.image_paths.append(path)

            elif self.attempt_reencode(path.path):
                if ImagePath(path.path).is_valid_image():
                    self.image_paths.append(ImagePath(path.path))
