import io
import json
import tarfile
import zipfile
from pathlib import Path
from PIL import Image
import requests

from config import (
    HEADERS,
    ITEM_LIST,
    get_archive_path,
    get_manifest_path,
    get_subset_dir,
)

def build_dataset(pack_nb: int = 80):
    manifest_path = get_manifest_path(pack_nb)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        packs = json.load(f)

    dataset_dir = get_subset_dir(pack_nb)
    dataset_dir.mkdir(parents=True, exist_ok=True)

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
                        category_dir = dataset_dir / item_category
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

    print(f"\nExtraction finished. Total items saved: {total_saved}")

def compress_dataset(pack_nb: int = 80):
    tar_path = get_archive_path(pack_nb)
    subset_dir = get_subset_dir(pack_nb)
    
    if not subset_dir.exists():
        print(f"Cannot compress, subset directory does not exist: {subset_dir}")
        return

    print(f"Compressing dataset subset {pack_nb}p to {tar_path}...")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(subset_dir, arcname=f"dataset_{pack_nb}p")
        
    print("Compression complete!")

if __name__ == "__main__":
    build_dataset(80)
    compress_dataset(80)