from .config import BETA_START, BETA_END
from .variance_scheduling import linear_scheduler, cosine_scheduler

from typing import Literal

import torch


class NoisingModel:
    def __init__(
        self, T, scheduler: Literal["linear", "cosine"], device: str = "cuda"
    ) -> None:
        self.device = device

        if scheduler == "linear":
            self.betas, self.sigmas, self.sigmas_bar = linear_scheduler(
                T=T, start=BETA_START, end=BETA_END
            )
        elif scheduler == "cosine":
            self.betas, self.sigmas, self.sigmas_bar = cosine_scheduler(
                T=T, start=BETA_START, end=BETA_END
            )

        self.betas = self.betas.to(self.device)
        self.sigmas = self.sigmas.to(self.device)
        self.sigmas_bar = self.sigmas_bar.to(self.device)

    def sample_at_step(self, batch, t):
        t = t - 1
        sigma_bar = self.sigmas_bar[t].view(-1, 1, 1, 1).to(self.device)

        epsilon = torch.randn_like(batch)

        mean = torch.sqrt(sigma_bar) * batch
        std = torch.sqrt(1 - sigma_bar) * epsilon

        x_t = mean + std
        return x_t, epsilon
