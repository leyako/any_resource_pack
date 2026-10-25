import hashlib
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_FOLDER = ROOT / "data"
DATASET_DIR = DATA_FOLDER / "dataset"

def deduplicate_item_folder(item_folder: Path) -> tuple[int, int]:
    """
    Removes exact duplicate PNG images within a single item directory based on MD5 hashing.
    Returns a tuple of (initial_count, duplicates_removed).
    """
    seen_hashes = set()
    initial_count = 0
    duplicates_removed = 0

    for img_path in list(item_folder.glob("*.png")):
        initial_count += 1
        with open(img_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        if file_hash in seen_hashes:
            img_path.unlink()  # Remove exact duplicate file
            duplicates_removed += 1
        else:
            seen_hashes.add(file_hash)

    return initial_count, duplicates_removed

def main():
    if not DATASET_DIR.exists():
        print(f"Dataset directory not found at: {DATASET_DIR}")
        return

    # Filter for subdirectories corresponding to item categories
    item_folders = [f for f in DATASET_DIR.iterdir() if f.is_dir()]
    
    if not item_folders:
        print("No item folders found in the dataset directory.")
        return

    print(f"Found {len(item_folders)} item categories in dataset. Running deduplication...\n")

    total_initial = 0
    total_removed = 0

    for folder in sorted(item_folders):
        initial, removed = deduplicate_item_folder(folder)
        remaining = initial - removed
        total_initial += initial
        total_removed += removed
        print(f"[{folder.name}] Initial: {initial} | Removed: {removed} | Clean Total: {remaining}")

    total_remaining = total_initial - total_removed
    print("\n--- Deduplication Summary ---")
    print(f"Total raw images processed: {total_initial}")
    print(f"Duplicates removed: {total_removed}")
    print(f"Total clean images remaining: {total_remaining}")

if __name__ == "__main__":
    main()