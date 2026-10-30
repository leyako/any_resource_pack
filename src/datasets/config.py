import os

from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

DATASET_NAME = "mc-items-32x"
DATASET_VERSION = "310p"

# Path stuff
ROOT = Path(__file__).parent.parent.parent
DATASET_PATH = ROOT / "dataset" / "data" / DATASET_NAME / DATASET_VERSION
TRAIN_PATH = DATASET_PATH / "train"
TEST_PATH = DATASET_PATH / "test"

HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN", None)

# Hyperparameters
BATCH_SIZE = 32
