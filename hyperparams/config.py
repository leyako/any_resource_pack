"""
Central configuration: every hyperparameter of the project lives here.

The per-package config.py files only re-export from this module, and the CLIs
use these values as their defaults.
"""

import os
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent

DATASET_NAME = "mc-items-32x"

DATASET_VERSION = "310p"

DATASET_PATH = ROOT / "dataset" / "data" / DATASET_NAME / DATASET_VERSION
TRAIN_PATH = DATASET_PATH / "train"
TEST_PATH = DATASET_PATH / "test"

IMG_SIZE = 32
BATCH_SIZE = 32
VAL_SPLIT = 0.2

GRAYSCALE = False

TIMESTEPS = 1000

SCHEDULER = "linear"

BETA_START = 1e-4
BETA_END = 0.02

# Width of the first feature map; doubled at each downsampling,
# so 64 -> 128 -> 256 -> 512 -> 1024 at the bottleneck.
BASE_CHANNELS = 64

TIME_EMB_DIM = 512

GROUP_NORM_GROUPS = 32

DROPOUT_RATE = 0.1

USE_SKIP = True

EPOCHS = 20
LEARNING_RATE = 2e-4
OPTIMIZER = "adam"

MODEL_PATH = ROOT / "src" / "training" / "models"
FIGURES_PATH = ROOT / "figures"

LATEST_CHECKPOINT = MODEL_PATH / "denoising-latest.pt"
RGBA_MODEL = MODEL_PATH / "rgba_model.pt"
GRAYSCALE_MODEL = MODEL_PATH / "grayscale_model.pt"

MODEL_PATH.mkdir(parents=True, exist_ok=True)
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN", None)


def get_device(preferred: str | None = None) -> str:
    """Picks the compute device: explicit choice, else CUDA, else MPS, else CPU."""
    if preferred is not None:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
