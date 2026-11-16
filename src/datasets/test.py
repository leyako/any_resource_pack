"""
Sanity check + report figure for the grayscale preprocessing pipeline.

    python3 -m src.datasets.test
"""

import matplotlib.pyplot as plt
import torch
from torchvision import datasets

from hyperparams.config import FIGURES_PATH
from src.datasets.config import TRAIN_PATH
from src.datasets.dataset_loader import load_dataset
from src.datasets.utils import denormalize


def show_grayscale_example(out_path=None, show: bool = True, verbose: bool = True):
    """Plots a raw RGBA sprite next to its grayscale+alpha transform."""
    # 1. Load a single raw RGBA image directly using PIL
    dataset = datasets.ImageFolder(TRAIN_PATH)
    raw_pil_img, label_idx = dataset[0]
    class_name = dataset.classes[label_idx]

    # 2. Get a batch from the grayscale pipeline
    gray_train_loader, _, _ = load_dataset(grayscale=True)
    images, _ = next(iter(gray_train_loader))  # Shape: [B, 2, 32, 32]

    # Select first transformed image from the batch and denormalize.
    # grayscale=True expands luminance back to R=G=B for display -> [4, 32, 32]
    gray_tensor = denormalize(images[0], grayscale=True)

    # Convert [C, H, W] PyTorch Tensor -> [H, W, C] NumPy array for Matplotlib
    gray_np = gray_tensor.permute(1, 2, 0).cpu().numpy()

    # 3. Plot side-by-side comparison
    _, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(raw_pil_img.convert("RGBA"))
    axes[0].set_title(f"Original RGBA\n({class_name})")
    axes[0].axis("off")

    axes[1].imshow(gray_np)
    axes[1].set_title("Stage 1: Grayscale RGBA\n[2, 32, 32] stored, R=G=B shown")
    axes[1].axis("off")

    plt.tight_layout()

    if out_path is None:
        out_path = FIGURES_PATH / "grayscale_pipeline.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved figure to {out_path}")

    if show:
        plt.show()
    plt.close()

    if verbose:
        r, g, b, a = gray_tensor[0], gray_tensor[1], gray_tensor[2], gray_tensor[3]
        print("--- Verification Check ---")
        print("Displayed shape:", gray_tensor.shape)
        print("Is R equal to G?", torch.allclose(r, g))
        print("Is G equal to B?", torch.allclose(g, b))
        print("Alpha min/max:", a.min().item(), "/", a.max().item())


if __name__ == "__main__":
    show_grayscale_example()
