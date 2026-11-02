import argparse
import sys
from config import MAX_PAGES
from create_dataset import build_dataset
from duplicate_handling import remove_duplicates
from find_packs import find_and_save_packs
from save_dataset import save_dataset_via_git


def run_pipeline(max_pages: int = MAX_PAGES):
    """
    Executes the complete dataset processing pipeline sequentially:
    1. Discovers and validates working resource pack URLs.
    2. Downloads packs and extracts targeted 32x32 item textures.
    3. Deduplicates visual duplicates using MD5 hashing.
    4. Uploads the processed dataset configuration to Hugging Face Hub.
    """
    print("=== Step 1/4: Discovering & Verifying Resource Packs ===")
    pack_nb = find_and_save_packs(max_pages=max_pages)

    if pack_nb == 0:
        print("Error: No working packs found. Aborting pipeline execution.")
        sys.exit(1)

    print(f"\n=== Step 2/4: Building Raw Dataset ({pack_nb}p subset) ===")
    build_dataset(pack_nb=pack_nb)

    print(f"\n=== Step 3/4: Handling Visual Duplicates ({pack_nb}p subset) ===")
    remove_duplicates(pack_number=pack_nb)

    print(f"\n=== Step 4/4: Uploading Dataset to HF (via Git) ({pack_nb}p subset) ===")
    save_dataset_via_git(pack_nb=pack_nb)

    print("\n=== Pipeline Execution Completed Successfully ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the end-to-end Minecraft 32x item dataset processing pipeline."
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=MAX_PAGES,
        help=f"Maximum search API pages to scrape (default: {MAX_PAGES}).",
    )

    args = parser.parse_args()
    run_pipeline(max_pages=args.max_pages)
