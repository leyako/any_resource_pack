import hashlib
from pathlib import Path
from config import get_subset_dir


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
    subset_dir = get_subset_dir(pack_number)

    for split in ["train", "test"]:
        split_dir = subset_dir / split
        if not split_dir.exists():
            continue

        item_folders = [f for f in split_dir.iterdir() if f.is_dir()]
        print(f"\n--- Deduplicating {split.upper()} split ({pack_number}p) ---")

        total_initial = 0
        total_removed = 0

        for folder in sorted(item_folders):
            initial, removed = deduplicate_item_folder(folder)
            total_initial += initial
            total_removed += removed
            print(
                f"[{folder.name}] Initial: {initial} | Removed: {removed} | Clean: {initial - removed}"
            )

        print(
            f"Summary for {split}: {total_initial - total_removed} clean images remaining."
        )
