import io
import json
import requests
import tarfile
import zipfile

from pathlib import Path
from PIL import Image

from config import ITEM_LIST

ROOT = Path(__file__).parent.parent
DATA_FOLDER = ROOT / "data"
DATASET_DIR = DATA_FOLDER / "dataset"

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}

def build_dataset():
    manifest_path = DATA_FOLDER / "80_packs_list.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        packs = json.load(f)

    total_saved = 0

    for idx, pack in enumerate(packs, start=1):
        slug = pack["slug"]
        url = pack["download_url"]
        
        try:
            res = requests.get(url, headers=HEADERS, timeout=15)
            if res.status_code != 200:
                continue

            # Read .mcpack in-memory as a zip archive
            with zipfile.ZipFile(io.BytesIO(res.content)) as z:
                for file_info in z.infolist():
                    filename = Path(file_info.filename).name
                    
                    if filename in ITEM_LIST:
                        item_category = Path(filename).stem  # ca renvoie "diamond_sword"
                        category_dir = DATASET_DIR / item_category
                        category_dir.mkdir(parents=True, exist_ok=True)

                        with z.open(file_info) as img_file:
                            img = Image.open(img_file)
                            
                            # Strictly keep 32x32 images
                            if img.size == (32, 32):
                                out_path = category_dir / f"{slug}_{filename}"
                                img.save(out_path)
                                total_saved += 1
                                
            print(f"[{idx}/{len(packs)}] Extracted assets from: {slug}")
            
        except Exception as e:
            print(f"Failed processing {slug}: {e}")

    print(f"\nExtraction finished. Total items saved: {total_saved}")

def compress_dataset():
    tar_path = DATA_FOLDER / "dataset_32x32.tar.gz"
    print(f"Compressing dataset to {tar_path}...")
    
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(DATASET_DIR, arcname="dataset")
        
    print("Compression complete!")

if __name__ == "__main__":
    build_dataset()
    compress_dataset()