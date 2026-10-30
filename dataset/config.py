import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

REPO_ID = "leyako/mc-items-32x"
API_URL = "https://api.texturepack.be/api/v1/search"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
DATASET_BASE_DIR = DATA_DIR / "mc-items-32x"

DATA_DIR.mkdir(parents=True, exist_ok=True)
DATASET_BASE_DIR.mkdir(parents=True, exist_ok=True)

MAX_PAGES = 20

def get_manifest_path(pack_nb: int) -> Path:
    """Returns path for a specific pack count manifest JSON."""
    return DATA_DIR / f"list_{pack_nb}p.json"

def get_subset_dir(pack_nb: int) -> Path:
    """Returns target dataset directory for a subset (e.g. data/mc-items-32x/80p)."""
    return DATASET_BASE_DIR / f"{pack_nb}p"

def get_archive_path(pack_nb: int) -> Path:
    """Returns tar.gz output path formatted by subset count."""
    return DATA_DIR / f"dataset_32x32_{pack_nb}p.tar.gz"

ITEM_LIST = {
    # Weapons & Tools
    "diamond_sword.png", "iron_sword.png", "gold_sword.png", "stone_sword.png",
    "diamond_pickaxe.png", "iron_pickaxe.png", "gold_pickaxe.png", "stone_pickaxe.png",
    "diamond_axe.png", "iron_axe.png", "gold_axe.png", "stone_axe.png",
    "diamond_shovel.png", "iron_shovel.png", "gold_shovel.png", "stone_shovel.png",
    "diamond_hoe.png", "iron_hoe.png", "gold_hoe.png", "stone_hoe.png",
    "bow_standby.png", "shears.png",
    
    # Armor Icons
    "diamond_helmet.png", "diamond_chestplate.png", "diamond_leggings.png", "diamond_boots.png",
    "iron_helmet.png", "iron_chestplate.png", "iron_leggings.png", "iron_boots.png",
    "gold_helmet.png", "gold_chestplate.png", "gold_leggings.png", "gold_boots.png",
    
    # Consumables & Food
    "apple.png", "apple_golden.png", "bread.png", "carrot.png", "potato.png",
    "ender_pearl.png", "mushroom_stew.png", "cookie.png"
}
