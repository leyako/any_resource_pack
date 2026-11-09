import itertools
import gc
import time
import logging
import traceback
import torch
import os
from pathlib import Path

from src.training.trainer import DenoisingTrainer
from src.datasets.dataset_loader import load_dataset


def setup_run_logger(run_name: str, log_dir: Path):
    """Sets up a dedicated file logger for a specific run."""
    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_dir / f"{run_name}.log")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(fh)

    return logger


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # setup experiment directories
    base_dir = Path("src/experiments")
    models_dir = base_dir / "models"
    logs_dir = base_dir / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # load dataset once to save time across runs
    train_loader, val_loader, _ = load_dataset(grayscale=True, val_split=0.2)

    # define the hyperparameter grid
    param_grid = {
        "epochs": [50, 75, 100],
        "lr": [1e-4, 2e-4, 5e-4],
        "T": [750, 1000],
        "optim": ["adam", "adamw"],
    }

    # Generate all combinations
    keys, values = zip(*param_grid.items())
    run_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Starting Grid Search: {len(run_configs)} total configurations.")

    for i, config in enumerate(run_configs):
        run_name = f"run_{i:03d}_ep{config['epochs']}_lr{config['lr']}_T{config['T']}_{config['optim']}_{config['scheduler']}"
        logger = setup_run_logger(run_name, logs_dir)

        logger.info(f"--- Starting {run_name} ---")
        logger.info(f"Config: {config}")
        print(f"[{i+1}/{len(run_configs)}] Training {run_name}...")

        start_time = time.time()

        try:
            # initialize trainer with current config
            trainer = DenoisingTrainer(
                train_loader=train_loader,
                val_loader=val_loader,
                optim=config["optim"],
                lr=config["lr"],
                T=config["T"],
                device=DEVICE,
            )

            # execute training loop
            final_val_loss = trainer.train(epochs=config["epochs"])

            training_time = time.time() - start_time
            logger.info(
                f"Completed in {training_time:.2f}s | Final Val Loss: {final_val_loss:.6f}"
            )

            # bundle metadata and weights
            checkpoint = {
                "model_state_dict": trainer.denoising_model.state_dict(),
                "metadata": {
                    "training_time_seconds": training_time,
                    "final_val_loss": final_val_loss,
                },
                "hyperparameters": config,
            }

            # save the bundled .pt file
            save_path = models_dir / f"{run_name}.pt"
            torch.save(checkpoint, save_path)
            logger.info(f"Saved checkpoint to {save_path}")

        except Exception as e:
            error_msg = traceback.format_exc()
            logger.error(f"RUN FAILED with exception:\n{error_msg}")
            print(f"  -> FAILED. See {logs_dir / run_name}.log for details.")

        finally:
            if "trainer" in locals():
                del trainer.denoising_model
                del trainer.optimizer
                del trainer.noising_model
                del trainer

            torch.cuda.empty_cache()
            gc.collect()

            logger.info("VRAM cleared. Moving to next run.\n")


if __name__ == "__main__":
    main()
