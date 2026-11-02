import torch


def denormalize(tensor: torch.Tensor, grayscale: bool = False) -> torch.Tensor:
    """Reverts [-1, 1] diffusion normalization back to [0, 1] range."""
    # Denormalize
    denom = (tensor * 0.5) + 0.5
    denom = torch.clamp(denom, 0.0, 1.0)

    if grayscale:
        # Extract Luminance and Alpha
        luminance = denom[0:1, :, :]  # [1, 32, 32]
        alpha = denom[1:2, :, :]  # [1, 32, 32]

        rgb = luminance.repeat(3, 1, 1)  # [3, 32, 32]

        # Recombine into [4, 32, 32] RGBA
        return torch.cat([rgb, alpha], dim=0)

    return denom
