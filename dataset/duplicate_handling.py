import hashlib
from pathlib import Path
from config import get_subset_dir
from create_dataset import compress_dataset

def deduplicate_item_folder(item_folder: Path) -> tuple[int, int]:
    seen_hashes = set()
    initial_count = 0
    duplicates_removed = 0

    for img_path in list(item_folder.glob("*.png")):
        initial_count += 1
        with open(img_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        if file_hash in seen_hashes:
            img_path.unlink()
            duplicates_removed += 1
        else:
            seen_hashes.add(file_hash)

    return initial_count, duplicates_removed

def remove_duplicates(pack_number: int = 80):
    dataset_dir = get_subset_dir(pack_number)

    if not dataset_dir.exists():
        print(f"Dataset directory not found at: {dataset_dir}")
        return

    item_folders = [f for f in dataset_dir.iterdir() if f.is_dir()]
    if not item_folders:
        print("No item folders found in the dataset directory.")
        return

    print(f"Running deduplication for {pack_number}p subset across {len(item_folders)} categories...\n")

    total_initial = 0
    total_removed = 0

    for folder in sorted(item_folders):
        initial, removed = deduplicate_item_folder(folder)
        remaining = initial - removed
        total_initial += initial
        total_removed += removed
        print(f"[{folder.name}] Initial: {initial} | Removed: {removed} | Clean Total: {remaining}")

    print("\n--- Deduplication Summary ---")
    print(f"Total raw images processed: {total_initial}")
    print(f"Duplicates removed: {total_removed}")
    print(f"Total clean images remaining: {total_initial - total_removed}")

if __name__ == "__main__":
    PACK_NB = 80
    remove_duplicates(PACK_NB)
    compress_dataset(PACK_NB)