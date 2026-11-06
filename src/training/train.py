import torch

from src.training.config import EPOCHS, TIMESTEPS
from src.training.trainer import DenoisingTrainer

from src.datasets.dataset_loader import load_dataset


def init_trainer():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, test_loader = load_dataset(grayscale=True)

    trainer = DenoisingTrainer(
        train_loader, val_loader, "adam", T=TIMESTEPS, device=device
    )

    return trainer


def main():
    trainer = init_trainer()
    trainer.train(EPOCHS)


if __name__ == "__main__":
    main()
