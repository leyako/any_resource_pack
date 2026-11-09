from .variance_scheduling import reverse_var_scheduler

import torch


class Sampler:
    def __init__(
        self, model, noising_model, img_size: int, channels: int, device: str
    ) -> None:
        self.T = noising_model.T
        self.betas = noising_model.betas
        self.sigmas = noising_model.sigmas
        self.sigmas_bar = noising_model.sigmas_bar
        self.reverse_variances = reverse_var_scheduler(self.betas)

        self.denoising_model = model
        self.device = device
        self.img_size = img_size
        self.channels = channels

    def generate(self, batch_size):
        x_t = torch.randn(
            batch_size, self.channels, self.img_size, self.img_size, device=self.device
        )

        for step in range(self.T, 0, -1):
            t = step - 1

            z = 0
            if step > 1:
                z = torch.randn_like(x_t)

            sigma = self.sigmas[t]
            sigma_bar = self.sigmas_bar[t]

            t_tensor = torch.full(
                (batch_size,), step, device=self.device, dtype=torch.long
            )
            noise_pred = self.denoising_model(x_t, t_tensor)

            mean = (
                1
                / torch.sqrt(sigma)
                * (x_t - (1 - sigma) / torch.sqrt(1 - sigma_bar) * noise_pred)
            )
            var = self.reverse_variances[t]

            x_t = mean + var * z

        #    if step % 500 == 0:
         #       print(f"Step {step} completed")

        return x_t
