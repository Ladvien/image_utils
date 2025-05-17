from PIL import Image as PILImage
import os


class ImageChecker:

    @staticmethod
    def is_valid_image(path: str) -> bool:
        if os.path.isdir(path):
            return False

        try:
            PILImage.open(path)
            return True
        except FileNotFoundError:
            print(f"File {path} not found. Maybe uppercase characters? Skipping...")
        except PILImage.UnidentifiedImageError as e:
            print(f"File {path} is not an image file. Skipping...")
            print(e)

        return False

    @staticmethod
    def can_be_reencoded(path: str) -> bool:
        try:
            image = PILImage.open(path)
            image.verify()
            return True
        except (IOError, PILImage.UnidentifiedImageError):
            print(f"File {path} cannot be re-encoded. Skipping...")
            return False

    @staticmethod
    def preview_image(path: str) -> None:
        try:
            image = PILImage.open(path)
            image.show()
        except (IOError, PILImage.UnidentifiedImageError):
            print(f"File {path} cannot be previewed. Skipping...")
