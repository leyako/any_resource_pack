"""
In this file, we propose different functions to schedule the variance of our noising process (the betas).

We have T steps, we want to have T corresponding variances scheduled for the forward process.
In our project, we will call sigma_t = 1 - beta_t what the DPPM paper calls alpha_t = 1 - beta_t

In short, we have betas and sigmas, as well as their cumulative product for each timestep t.
"""

import numpy as np
import torch


def linear_scheduler(T=1000, start=1e-4, end=0.02):
    """
    Linear scheduler that follows the `Experiments` discussed in the DDPM paper.

    The default values used by the authors of the paper is 10^-4 for the starting beta, and 0.02 for the ending beta.

    The goal is to create a linear scheduler that does not noise too fast our images, since we are working with low-res
    32x32 images.

    Thanks to an already existing repo (https://github.com/ThiagoLira/ToyDiffusion/blob/main/DiffusionGenealogy/ddpm/utils.py),
    we were able to understand and include square root linear interpolation for the linear space that our betas will follow.

    Having the sqrt of _start_ and _end_ will increase their values (since they are < 1), we will then be able to create
    our linear space with a smaller increase at each step. This allows earlier values of the schedule to grow slowly and
    allows our images to keep more information in earlier steps.
    """

    betas = np.linspace(start**0.5, end**0.5, T) ** 2  # Sqrt-linear space
    betas = torch.tensor(betas, dtype=torch.float32)

    sigmas = 1 - betas
    sigmas_bar = torch.cumprod(sigmas, dim=0)

    return betas, sigmas, sigmas_bar


def cosine_scheduler(T=1000, start=1e-4, end=0.02, s=0.008):
    """Cosine scheduler explained in the "Improved DDPM" paper (Nichol & Dhariwal, 2021)

    This scheduler avoids the abrupt changes in noise level. It allows to have a linear drop-off
    in the middle values while not changing too much in the extreme values (t = 0, t = T).

    The s is a little offset that allows the beta values to not be too close to 0 on small steps (t = 0).
    The authors found that having tiny amounts of noise makes it hard for the network to predict epsilon
    accurately enough.
    """
    steps = np.arange(T + 1, dtype=np.float64)

    f = np.cos((steps / T + s) / (1 + s) * np.pi / 2) ** 2

    sigmas_bar = f / f[0]

    # [1:] takes from t = 1, and [:-1] takes from t = 0 up to t = T - 1
    betas = np.clip(1 - sigmas_bar[1:] / sigmas_bar[:-1], 0, 0.999)
    betas = torch.tensor(betas, dtype=torch.float32)

    sigmas = 1 - betas
    sigmas_bar = torch.cumprod(sigmas, dim=0)

    return betas, sigmas, sigmas_bar
