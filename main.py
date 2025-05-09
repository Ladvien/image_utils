import os
import shutil
from pathlib import Path

# --- CONFIGURATION ---

SOURCE_DIR = Path("/Volumes/T7/images/tmp")
DEST_DIR = Path("/Volumes/T7/images/structured")
FILES_PER_FOLDER = 2000  # Max files per terminal directory
DRY_RUN = False  # Change to False to actually move files

# --- SCRIPT ---


def get_nested_folder(index):
    """Two-level folder structure based on index."""
    level1 = str(index // 1000000).zfill(2)  # Up to 99
    level2 = str((index // 10000) % 100).zfill(2)  # Up to 99
    return level1, level2


def get_existing_file_count(dest_dir):
    return sum(1 for _ in dest_dir.rglob("*") if _.is_file())


def move_files_to_nested(source_dir, dest_dir, batch_size):
    count = get_existing_file_count(dest_dir)

    for file_path in source_dir.rglob("*"):
        if not file_path.is_file():
            continue

        # Skip if already moved
        already_moved = False
        for root, _, files in os.walk(dest_dir):
            if file_path.name in files:
                already_moved = True
                break
        if already_moved:
            print(f"Skipping {file_path.name} — already exists in destination.")
            continue

        level1, level2 = get_nested_folder(count)
        dest_folder = dest_dir / level1 / level2
        dest_folder.mkdir(parents=True, exist_ok=True)

        dest_path = dest_folder / file_path.name

        if dest_path.exists():
            print(f"Skipping {file_path.name} — already exists at {dest_path}.")
        else:
            if DRY_RUN:
                print(f"[DRY RUN] Would move: {file_path} -> {dest_path}")
            else:
                shutil.move(str(file_path), str(dest_path))
                print(f"Moved: {file_path} -> {dest_path}")
            count += 1

            if count % batch_size == 0:
                print(f"Processed {count} files...")

    print(f"✅ Finished processing {count} files.")


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    move_files_to_nested(SOURCE_DIR, DEST_DIR, FILES_PER_FOLDER)


if __name__ == "__main__":
    main()
