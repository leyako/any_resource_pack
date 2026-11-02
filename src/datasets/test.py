from dataset_loader import load_dataset
from torchvision import datasets
from config import TRAIN_PATH

import matplotlib.pyplot as plt
import torch


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Reverts [-1, 1] diffusion normalization back to [0, 1] range."""
    # Denormalize
    denom = (tensor * 0.5) + 0.5
    denom = torch.clamp(denom, 0.0, 1.0)

    # Extract Luminance and Alpha
    luminance = denom[0:1, :, :]  # [1, 32, 32]
    alpha = denom[1:2, :, :]  # [1, 32, 32]

    rgb = luminance.repeat(3, 1, 1)  # [3, 32, 32]

    # Recombine into [4, 32, 32] RGBA
    return torch.cat([rgb, alpha], dim=0)


def show_grayscale_example():
    # 1. Load a single raw RGBA image directly using PIL
    dataset = datasets.ImageFolder(TRAIN_PATH)
    raw_pil_img, label_idx = dataset[0]
    class_name = dataset.classes[label_idx]

    # 2. Get a batch from the grayscale pipeline
    gray_train_loader, _ = load_dataset(grayscale=True)
    images, _ = next(iter(gray_train_loader))  # Shape: [B, 4, 32, 32]

    # Select first transformed image from the batch and denormalize
    gray_tensor = denormalize(images[0])  # Shape: [4, 32, 32] in [0, 1]

    # Convert [C, H, W] PyTorch Tensor -> [H, W, C] NumPy array for Matplotlib
    gray_np = gray_tensor.permute(1, 2, 0).cpu().numpy()

    # 3. Plot side-by-side comparison
    _, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Raw Color Input
    axes[0].imshow(raw_pil_img.convert("RGBA"))
    axes[0].set_title(f"Original RGBA\n({class_name})")
    axes[0].axis("off")

    # Grayscale RGBA Output
    axes[1].imshow(gray_np)
    axes[1].set_title("Stage 1: Grayscale RGBA\n[4, 32, 32] (R=G=B)")
    axes[1].axis("off")

    plt.tight_layout()
    # plt.savefig("grayscale_example.png", bbox_inches="tight", dpi=150)
    plt.show()

    # 4. Verify tensor properties in console
    r, g, b, a = gray_tensor[0], gray_tensor[1], gray_tensor[2], gray_tensor[3]
    print("--- Verification Check ---")
    print("Tensor Shape:", gray_tensor.shape)
    print("Is R equal to G?", torch.allclose(r, g))
    print("Is G equal to B?", torch.allclose(g, b))
    print("Alpha min/max:", a.min().item(), "/", a.max().item())


if __name__ == "__main__":
    show_grayscale_example()
