"""
Reading and writing PNG files as tensors, without any torchvision dependency.

Kept separate from dataset_loader so that noising an arbitrary image needs only
torch and pillow.
"""

import numpy as np
import torch
from PIL import Image


class RGBAToGrayscale:
    """
    Converts RGBA Tensor [4, 32, 32] to Grayscale RGBA [2, 32, 32].
    Computes luminance across RGB.
    Alpha channel remains completely untouched.
    """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor is [4, 32, 32] in range [0.0, 1.0]
        rgb = tensor[:3, :, :]
        alpha = tensor[3:, :, :]

        # Calculate luminance (Y = 0.299 R + 0.587 G + 0.114 B)
        weights = torch.tensor([0.299, 0.587, 0.114], device=tensor.device).view(
            3, 1, 1
        )
        gray = (rgb * weights).sum(dim=0, keepdim=True)  # Shape: [1, 32, 32]

        # Recombine with Alpha -> Shape: [2, 32, 32]
        return torch.cat([gray, alpha], dim=0)


def load_image(path, grayscale: bool = True) -> torch.Tensor:
    """
    Loads a PNG as a [C, H, W] tensor normalised to [-1, 1].

    Mirrors the dataset transforms minus the training-only jitter, so that an
    external image sits in the range the variance schedule was calibrated for.
    """
    img = Image.open(path).convert("RGBA")

    # [H, W, 4] uint8 -> [4, H, W] float in [0, 1]
    array = np.asarray(img, dtype=np.uint8)
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).float() / 255.0

    if grayscale:
        tensor = RGBAToGrayscale()(tensor)

    return (tensor - 0.5) / 0.5


def to_pil(tensor: torch.Tensor, grayscale: bool = True) -> Image.Image:
    """Turns a single [C, H, W] tensor in [-1, 1] back into an RGBA image."""
    from .utils import denormalize

    rgba = denormalize(tensor.detach().cpu(), grayscale=grayscale)
    array = (rgba.permute(1, 2, 0) * 255).round().to(torch.uint8).numpy()
    return Image.fromarray(array, mode="RGBA")
