"""
Forward process: noises an image at the given timesteps.

    python -m src.noising.noise --t 0 100 300 600 1000
    python -m src.noising.noise --image path/to/sprite.png --t 500

Uses the closed form of q(x_t | x_0), so any t costs a single operation.
Rendering a strip of increasing t shows how fast a schedule destroys the image.
"""

import argparse

import matplotlib.pyplot as plt
import torch

from hyperparams.config import (
    FIGURES_PATH,
    SCHEDULER,
    TIMESTEPS,
    get_device,
    set_seed,
)
from src.datasets.dataset_loader import get_transforms, load_dataset, rgba_loader
from src.datasets.utils import denormalize
from src.noising.noising_model import NoisingModel


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Apply the DDPM forward process to an image at given timesteps."
    )
    parser.add_argument(
        "--t",
        type=int,
        nargs="+",
        default=[0, 100, 250, 500, 750, 1000],
        help="Timesteps to render. 0 means the untouched image.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="A PNG to noise. Defaults to one image from the training set.",
    )
    parser.add_argument(
        "--index", type=int, default=0, help="Which image of the batch to use."
    )
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS, help="T")
    parser.add_argument(
        "--scheduler", choices=["linear", "cosine"], default=SCHEDULER
    )
    parser.add_argument(
        "--rgba", action="store_true", help="Keep full colour instead of grayscale."
    )
    parser.add_argument("--out", default=None, help="Output PNG.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args(argv)


def load_single_image(args, grayscale: bool):
    """
    Returns one image as a [1, C, H, W] tensor normalised to [-1, 1].

    An external PNG goes through the dataset transforms (minus the training-only
    jitter), otherwise its value range would not match the schedule.
    """
    if args.image is not None:
        transform = get_transforms(grayscale=grayscale, is_train=False)
        return transform(rgba_loader(args.image)).unsqueeze(0)

    train_loader, _, _ = load_dataset(grayscale=grayscale)
    images, _ = next(iter(train_loader))
    return images[args.index : args.index + 1]


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)

    device = get_device(args.device)
    grayscale = not args.rgba

    image = load_single_image(args, grayscale)

    noising_model = NoisingModel(
        T=args.timesteps, scheduler=args.scheduler, device=device
    )

    steps = sorted(args.t)
    _, axes = plt.subplots(1, len(steps), figsize=(len(steps) * 1.8, 2.2))
    if len(steps) == 1:
        axes = [axes]

    for ax, t in zip(axes, steps):
        # t = 0 is x_0 itself; the schedule starts at t = 1.
        if t <= 0:
            noised = image.to(device)
        else:
            t_tensor = torch.full((1,), t, dtype=torch.long)
            noised, _ = noising_model.sample_at_step(batch=image, t=t_tensor)

        img = denormalize(noised.cpu(), grayscale=grayscale)[0]
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(f"t = {t}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()

    out_path = args.out or FIGURES_PATH / "noising_process.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved figure to {out_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
