import torch

from config import EPOCHS, TIMESTEPS
from src.datasets.dataset_loader import load_dataset
from trainer import DenoisingTrainer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, test_loader = load_dataset(grayscale=True)

    trainer = DenoisingTrainer(
        train_loader, val_loader, "adam", T=TIMESTEPS, device=device
    )

    trainer.train(EPOCHS)


if __name__ == "__main__":
    main()
