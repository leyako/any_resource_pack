import json
import concurrent.futures
from pathlib import Path
import requests

ROOT = Path(__file__).parent.parent
DATA_FOLDER = ROOT / "data"
DATA_FOLDER.mkdir(parents=True, exist_ok=True)

MAX_PAGES = 5

API_URL = "https://api.texturepack.be/api/v1/search"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
}

def check_candidate_url(candidate: dict) -> dict | None:
    """Tests if candidate static CDN URLs return an HTTP 200 OK status."""
    for url in candidate["urls"]:
        try:
            res = requests.head(url, headers=HEADERS, allow_redirects=True, timeout=4)
            if res.status_code == 200:
                return {
                    "id": candidate["id"],
                    "name": candidate["name"],
                    "slug": candidate["slug"],
                    "downloads": candidate["downloads"],
                    "tags": candidate["tags"],
                    "download_url": url
                }
        except requests.RequestException:
            continue
    return None

def fetch_pack_manifest(max_pages=20):
    """
    Queries the texturepack.be search API, verifies candidate download links concurrently,
    and returns a list of all verified working packs across all pages.
    """
    candidates = []
    
    for page in range(1, max_pages + 1):
        params = {
            "page": page,
            "limit": 20,
            "resolution": "32x32",
            "sortBy": "download_count",
            "sortOrder": "desc"
        }
        
        response = requests.get(API_URL, headers=HEADERS, params=params)
        if response.status_code != 200:
            print(f"Error fetching page {page}: HTTP {response.status_code}")
            break
            
        payload = response.json()
        results = payload.get("data", {}).get("results", [])
        
        if not results:
            break
            
        for pack in results:
            slug = pack.get("slug")
            name = pack.get("name") or pack.get("packName")
            downloads = pack.get("download_count")
            tags = pack.get("tags", [])
            
            candidates.append({
                "id": pack.get("id"),
                "name": name,
                "slug": slug,
                "downloads": downloads,
                "tags": tags,
                "urls": [
                    f"https://sdn.texturepack.be/packs/{slug}/{slug}.mcpack",
                ]
            })
            
        pagination = payload.get("data", {}).get("pagination", {})
        if not pagination.get("hasNext", False):
            break

    print(f"Retrieved {len(candidates)} candidates from API. Testing working direct URLs...")
    working_manifest = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_candidate = {
            executor.submit(check_candidate_url, c): c for c in candidates
        }
        
        for future in concurrent.futures.as_completed(future_to_candidate):
            result = future.result()
            if result:
                working_manifest.append(result)
                print(f"[{len(working_manifest)}] Confirmed: {result['name']}")

    return working_manifest

if __name__ == "__main__":
    packs = fetch_pack_manifest(max_pages=MAX_PAGES)
    
    print(f"\nSuccessfully verified {len(packs)} working packs in total.")
    
    output_path = DATA_FOLDER / f"{len(packs)}_packs_list.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(packs, f, indent=2, ensure_ascii=False)
        
    print(f"Saved manifest to {output_path}")