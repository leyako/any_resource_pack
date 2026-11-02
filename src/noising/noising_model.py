from variance_scheduling import linear_scheduler, cosine_scheduler

from typing import Literal
from config import BETA_START, BETA_END

import torch


class NoisingModel:
    def __init__(self, T, x_0, scheduler: Literal["linear", "cosine"]) -> None:
        self.x_0 = x_0

        if scheduler == "linear":
            self.betas, self.sigmas, self.sigmas_bar = linear_scheduler(
                T=T, start=BETA_START, end=BETA_END
            )
        elif scheduler == "cosine":
            self.betas, self.sigmas, self.sigmas_bar = cosine_scheduler(
                T=T, start=BETA_START, end=BETA_END
            )

    def sample_at_step(self, t):
        mean = torch.sqrt(self.sigmas_bar[t]) * self.x_0
        std = 1 - self.sigmas_bar[t] * torch.eye(self.x_0.size[0])
        res = torch.normal(mean, std)

        return res
