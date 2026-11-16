import argparse
from pathlib import Path

from hyperparams.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_PATH,
    OPTIMIZER,
    SCHEDULER,
    TIMESTEPS,
    VAL_SPLIT,
    get_device,
    set_seed,
)
from src.datasets.dataset_loader import load_dataset
from src.training.trainer import DenoisingTrainer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the DDPM denoising UNet on Minecraft item textures."
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS, help="T")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--scheduler", choices=["linear", "cosine"], default=SCHEDULER)
    parser.add_argument("--optim", choices=["adam", "adamw"], default=OPTIMIZER)
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT)
    parser.add_argument(
        "--rgba",
        action="store_true",
        help="Train on 4-channel RGBA instead of the default grayscale+alpha.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=MODEL_PATH,
        help="Directory where checkpoints (.pt) will be saved.",
    )
    parser.add_argument(
        "--save-epochs",
        type=int,
        nargs="*",
        default=None,
        help="Optional intermediate epochs to save (final epoch is always saved).",
    )
    parser.add_argument(
        "--device", default=None, help="cuda | mps | cpu (auto-detected by default)"
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def init_trainer(args):
    device = get_device(args.device)
    print(f"Device: {device}")

    train_loader, val_loader, _ = load_dataset(
        grayscale=not args.rgba,
        val_split=args.val_split,
        batch_size=args.batch_size,
    )

    return DenoisingTrainer(
        train_loader,
        val_loader,
        optim=args.optim,
        T=args.timesteps,
        device=device,
        lr=args.lr,
        scheduler=args.scheduler,
    )


def main(argv=None):
    args = parse_args(argv)

    if args.seed is not None:
        set_seed(args.seed)

    trainer = init_trainer(args)
    trainer.train(
        epochs=args.epochs,
        save_epochs=args.save_epochs,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
