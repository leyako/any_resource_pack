import gc
import time
import traceback
import torch
from pathlib import Path

from src.training.trainer import DenoisingTrainer
from src.datasets.dataset_loader import load_dataset


def train_models():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    base_dir = Path(__file__).parent
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # load dataset
    train_loader, val_loader, _ = load_dataset(grayscale=True, val_split=0.2)

    best_configs = [
        {"lr": 5e-4, "T": 750, "optim": "adamw"},
        {"lr": 1e-4, "T": 750, "optim": "adamw"},
        {"lr": 2e-4, "T": 750, "optim": "adam"},
        {"lr": 5e-4, "T": 1000, "optim": "adamw"},
        {"lr": 2e-4, "T": 750, "optim": "adamw"},
        {"lr": 1e-4, "T": 1000, "optim": "adamw"},
        {"lr": 5e-4, "T": 750, "optim": "adam"},
        {"lr": 1e-4, "T": 750, "optim": "adam"},
        {"lr": 2e-4, "T": 1000, "optim": "adamw"},
        {"lr": 1e-4, "T": 1000, "optim": "adam"},
    ]

    TOTAL_EPOCHS = 200
    SAVE_EPOCHS = [100, 125, 150, 175, 200]

    for i, config in enumerate(best_configs):
        print(
            f"\n[{i+1}/{len(best_configs)}] Starting Top 10 tuning for config: {config}"
        )

        # build the full config for the evaluation script
        full_config = config.copy()

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
            # strict VRAM cleanup
            if "trainer" in locals():
                del trainer.denoising_model
                del trainer.optimizer
                del trainer.noising_model
                del trainer

            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    train_models()
