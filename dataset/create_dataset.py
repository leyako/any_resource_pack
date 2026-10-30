import io
import json
import random
import requests
import zipfile
from pathlib import Path
from PIL import Image

from config import HEADERS, ITEM_LIST, get_manifest_path, get_subset_dir


def build_dataset(pack_nb: int = 80, test_ratio: float = 0.2, seed: int = 42):
    manifest_path = get_manifest_path(pack_nb)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        packs = json.load(f)

    subset_dir = get_subset_dir(pack_nb)
    train_dir = subset_dir / "train"
    test_dir = subset_dir / "test"

    random.seed(seed)
    total_saved = 0

    for idx, pack in enumerate(packs, start=1):
        slug = pack["slug"]
        url = pack["download_url"]

        try:
            res = requests.get(url, headers=HEADERS, timeout=15)
            if res.status_code != 200:
                continue

            with zipfile.ZipFile(io.BytesIO(res.content)) as z:
                for file_info in z.infolist():
                    filename = Path(file_info.filename).name

                    if filename in ITEM_LIST:
                        item_category = Path(filename).stem

                        # Assign to train or test split based on ratio
                        split_dir = (
                            test_dir if random.random() < test_ratio else train_dir
                        )
                        category_dir = split_dir / item_category
                        category_dir.mkdir(parents=True, exist_ok=True)

                        with z.open(file_info) as img_file:
                            img = Image.open(img_file)
                            if img.size == (32, 32):
                                out_path = category_dir / f"{slug}_{filename}"
                                img.save(out_path)
                                total_saved += 1

            print(f"[{idx}/{len(packs)}] Extracted assets from: {slug}")

        except Exception as e:
            print(f"Failed processing {slug}: {e}")

    print(f"\nExtraction finished. Total items saved across train/test: {total_saved}")
