import gc
import time
import traceback
import itertools
import torch
from pathlib import Path

from src.training.trainer import DenoisingTrainer
from src.datasets.dataset_loader import load_dataset


def rgba_grid_search():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    base_dir = Path("src/experiments")
    models_dir = base_dir / "rgba_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load 4-channel dataset
    train_loader, val_loader, _ = load_dataset(grayscale=False, val_split=0.2)

    param_grid = {
        "lr": [1e-4, 2e-4, 3e-4],
        "T": [750, 1000],
        "optim": ["adamw"],
        "epochs": [100, 150, 200],
    }

    keys = ["lr", "T", "optim"]
    unique_configs = [
        dict(zip(keys, v))
        for v in itertools.product(
            param_grid["lr"], param_grid["T"], param_grid["optim"]
        )
    ]

    SAVE_EPOCHS = param_grid["epochs"]
    TOTAL_EPOCHS = SAVE_EPOCHS[-1]

    for i, config in enumerate(unique_configs):
        print(f"\n[{i+1}/{len(unique_configs)}] Starting RGBA Run: {config}")

        full_config = config.copy()
        full_config["scheduler"] = "linear"
        full_config["channels"] = 4

        try:
            trainer = DenoisingTrainer(
                train_loader=train_loader,
                val_loader=val_loader,
                optim=config["optim"],
                lr=config["lr"],
                T=config["T"],
                device=DEVICE,
            )

            trainer.train(
                epochs=TOTAL_EPOCHS,
                save_epochs=SAVE_EPOCHS,
                config=full_config,
                save_dir=models_dir,
            )

        except Exception as e:
            print(f"FAILED run {i+1}:\n{traceback.format_exc()}")

        finally:
            if "trainer" in locals():
                del trainer.denoising_model
                del trainer.optimizer
                del trainer.noising_model
                del trainer

            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    rgba_grid_search()
