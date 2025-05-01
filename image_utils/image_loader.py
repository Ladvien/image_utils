from glob import glob
from pathlib import Path
from typing import Iterable, List
from PIL import Image as PILImage
import random

from .image_path import ImagePath


class ImageLoader:
    def __init__(
        self, input_folder: str, extensions: List[str] = None, shuffle: bool = False
    ):
        self.input_folder = input_folder
        self.shuffle = shuffle
        if extensions:
            self.extensions = [ext.lower() for ext in extensions]
        else:
            self.extensions = [".jpg", ".jpeg", ".png"]

        self.__setup(input_folder)

    def __iter__(self):
        """Return self as an iterator."""
        return self

    def __next__(self) -> ImagePath:
        """Return the next ImagePath."""
        if self._index >= len(self.image_paths):
            raise StopIteration
        result = self.image_paths[self._index]
        self._index += 1
        return result

    def __getitem__(self, index: int) -> ImagePath:
        """Allow index access to ImagePath objects."""
        return self.image_paths[index]

    def __len__(self) -> int:
        """Allow len(loader) to return number of images."""
        return len(self.image_paths)

    def iter_image_paths(self) -> Iterable[ImagePath]:
        """Return a new iterator over image paths."""
        return iter(self.image_paths)

    def iter_images(self) -> Iterable[PILImage.Image]:
        """Yield all loaded PIL Images."""
        return (img_path.load() for img_path in self.image_paths)

    def get_all_images(self) -> List[PILImage.Image]:
        """Return all images eagerly."""
        return [image_path.load() for image_path in self.image_paths]

    def get_all_image_paths(self) -> List[str]:
        """Return all file paths eagerly."""
        return [image_path.path for image_path in self.image_paths]

    def get_image(self, index: int) -> PILImage.Image:
        """Return a loaded image by index."""
        if index < 0 or index >= len(self.image_paths):
            raise IndexError("Index out of range.")
        return self.image_paths[index].load()

    def get_image_path(self, index: int) -> ImagePath:
        """Return an ImagePath by index."""
        return self.image_paths[index]

    def total(self) -> int:
        """Return the total number of images."""
        return len(self.image_paths)

    def reset(self):
        """Reset the iterator to the beginning."""
        self.__setup(self.input_folder)

    def __setup(self, input_folder: str):
        """Reset the iterator to the beginning."""
        raw_image_paths = list(Path(input_folder).rglob("*", case_sensitive=False))

        if not raw_image_paths:
            raise Exception(f"No files found in '{self.input_folder}'.")

        filtered_image_paths = [
            path for path in raw_image_paths if path.suffix.lower() in self.extensions
        ]

        if not filtered_image_paths:
            raise Exception(
                f"No files found in '{self.input_folder}' with extensions {self.extensions}."
            )

        if self.shuffle:
            random.shuffle(filtered_image_paths)
        else:
            filtered_image_paths.sort()

        potential_image_paths = [ImagePath(path) for path in filtered_image_paths]

        self.image_paths = [
            path for path in potential_image_paths if path.is_valid_image()
        ]
        self._index = 0
