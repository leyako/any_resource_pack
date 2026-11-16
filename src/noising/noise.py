"""
Forward process: noises 3 random dataset sprites at 7 timesteps.

    python3 -m src.noising.noise

Uses the closed form of q(x_t | x_0), so any t costs a single operation. Needs
only torch and pillow. Every run draws different sprites.
"""

import random

import matplotlib.pyplot as plt
import torch

from hyperparams.config import (
    DATASET_PATH,
    FIGURES_PATH,
    GRAYSCALE,
    SCHEDULER,
    TIMESTEPS,
    get_device,
)
from src.datasets.image_io import load_image
from src.datasets.utils import denormalize
from src.noising.noising_model import NoisingModel

N_IMAGES = 3
N_STEPS = 7


def main():
    device = get_device()

    # N_STEPS points evenly spaced over [0, T], endpoints included.
    steps = [round(i * TIMESTEPS / (N_STEPS - 1)) for i in range(N_STEPS)]

    sprites = sorted(DATASET_PATH.rglob("*.png"))
    if len(sprites) < N_IMAGES:
        raise FileNotFoundError(
            f"Found {len(sprites)} sprites under {DATASET_PATH}, need {N_IMAGES}.\n"
            "Fetch the dataset first: git submodule update --init --recursive"
        )

    paths = random.sample(sprites, N_IMAGES)
    for p in paths:
        print(f"  picked {p}")

    model = NoisingModel(T=TIMESTEPS, scheduler=SCHEDULER, device=device)

    _, axes = plt.subplots(
        N_IMAGES, N_STEPS, figsize=(N_STEPS * 1.5, N_IMAGES * 1.65), squeeze=False
    )

    for row, path in enumerate(paths):
        batch = load_image(path, GRAYSCALE).unsqueeze(0)

        for col, t in enumerate(steps):
            # t = 0 is x_0 itself; the schedule starts at t = 1.
            if t <= 0:
                noised = batch.to(device)
            else:
                t_tensor = torch.full((1,), t, dtype=torch.long)
                noised, _ = model.sample_at_step(batch=batch, t=t_tensor)

            rgba = denormalize(noised.cpu(), grayscale=GRAYSCALE)[0]

            ax = axes[row][col]
            ax.imshow(rgba.permute(1, 2, 0).numpy())
            ax.axis("off")
            if row == 0:
                ax.set_title(f"t = {t}", fontsize=9)
            if col == 0:
                # Label by item class: the filename also carries the pack name.
                ax.text(
                    -0.08, 0.5, path.parent.name, transform=ax.transAxes,
                    rotation=90, va="center", ha="right", fontsize=8,
                )

    plt.tight_layout()
    out_path = FIGURES_PATH / "noising_process.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved grid to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
