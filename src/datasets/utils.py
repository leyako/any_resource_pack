import torch
import matplotlib.pyplot as plt
import math

from .config import ROOT


def denormalize(tensor: torch.Tensor, grayscale: bool = False) -> torch.Tensor:
    """Reverts [-1, 1] diffusion normalization back to [0, 1] range."""
    # Denormalize
    denom = (tensor * 0.5) + 0.5
    denom = torch.clamp(denom, 0.0, 1.0)

    if grayscale:
        # Extract Luminance and Alpha
        luminance = denom[..., 0:1, :, :]  # [B, 1, 32, 32]
        alpha = denom[..., 1:2, :, :]  # [B, 1, 32, 32]

        return torch.cat([luminance, luminance, luminance, alpha], dim=-3)

    return denom


def show_image(
    tensor, slug: str = "", grayscale: bool = False, out_path=None, show: bool = True
):
    tensor = tensor.detach().cpu()

    # single images become a batch of 1
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    if tensor.dim() != 4:
        raise ValueError(f"Expected 3D or 4D tensor, got {tensor.dim()}D tensor.")

    # denormalize the batch
    tensor = denormalize(tensor, grayscale=grayscale)

    # dynamic grid plotting
    b_size = tensor.size(0)
    cols = min(b_size, 8)
    rows = math.ceil(b_size / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))

    # Force axes into a flat list even if it's just 1 item, so we can always iterate over it
    if b_size == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(b_size):
        # convert [C, H, W] -> [H, W, C]
        np_img = tensor[i].permute(1, 2, 0).numpy()
        axes[i].imshow(np_img)
        axes[i].axis("off")

    # Hide any empty subplots at the end of the grid
    for j in range(b_size, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if out_path is None:
        out_path = ROOT / "figures" / f"{slug}_{b_size}_images.png"

    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved figure to {out_path}")

    if show:
        plt.show()
    plt.close(fig)
