import json
import concurrent.futures
import requests

from config import API_URL, HEADERS, get_manifest_path

MAX_PAGES = 20

def check_candidate_url(candidate: dict) -> dict | None:
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

def fetch_pack_manifest(resolution: str = "32x32", max_pages: int = 20):
    candidates = []
    
    for page in range(1, max_pages + 1):
        params = {
            "page": page,
            "limit": 20,
            "resolution": resolution,
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
            candidates.append({
                "id": pack.get("id"),
                "name": pack.get("name") or pack.get("packName"),
                "slug": slug,
                "downloads": pack.get("download_count"),
                "tags": pack.get("tags", []),
                "urls": [f"https://sdn.texturepack.be/packs/{slug}/{slug}.mcpack"]
            })
            
        if not payload.get("data", {}).get("pagination", {}).get("hasNext", False):
            break

    working_manifest = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_candidate = {executor.submit(check_candidate_url, c): c for c in candidates}
        for future in concurrent.futures.as_completed(future_to_candidate):
            result = future.result()
            if result:
                working_manifest.append(result)

    return working_manifest

def find_and_save_packs(max_pages: int = MAX_PAGES) -> int:
    packs = fetch_pack_manifest(max_pages=max_pages)
    pack_nb = len(packs)

    # Taking the lowest multiple of 10
    pack_nb = pack_nb // 10  * 10
    
    output_path = get_manifest_path(pack_nb)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(packs, f, indent=2, ensure_ascii=False)
        
    print(f"Verified {pack_nb} packs. Saved manifest to {output_path}")
    return pack_nb

if __name__ == "__main__":
    find_and_save_packs(MAX_PAGES)