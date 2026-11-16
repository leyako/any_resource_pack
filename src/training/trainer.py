from typing import Literal
from tqdm import tqdm

from src.denoising.unet import DenoisingUNet
from src.noising.noising_model import NoisingModel

from hyperparams.config import (
    BASE_CHANNELS,
    LATEST_CHECKPOINT,
    LEARNING_RATE,
    MODEL_PATH,
    SCHEDULER,
)

import torch
import torch.nn.functional as F


class DenoisingTrainer:
    def __init__(
        self,
        train_loader,
        val_loader,
        optim: Literal["adam", "adamw"],
        T: int,
        device: str = "cuda",
        scheduler: Literal["linear", "cosine"] = SCHEDULER,
        lr: float = LEARNING_RATE,
        base: int = BASE_CHANNELS,
    ) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.T = T
        self.device = device

        batch, _ = next(iter(train_loader))
        self.denoising_model = DenoisingUNet(
            batch.size(1), batch.size(1), base=base, T=T
        ).to(self.device)

        if optim == "adam":
            self.optimizer = torch.optim.Adam(
                self.denoising_model.parameters(), lr=lr
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.denoising_model.parameters(), lr=lr
            )

        self.noising_model = NoisingModel(
            T=T, scheduler=scheduler, device=self.device
        )

    def train_step(self, batch):
        batch = batch.to(self.device)
        b_size = batch.size(0)

        # Sample random timesteps
        timesteps = torch.randint(1, self.T + 1, (b_size,))
        x_ts, epsilons = self.noising_model.sample_at_step(batch=batch, t=timesteps)

        self.optimizer.zero_grad()
        pred_epsilons = self.denoising_model(x_ts, timesteps)

        loss = F.mse_loss(pred_epsilons, epsilons)
        loss.backward()
        self.optimizer.step()

        return loss.cpu().item()

    def validate(self):
        self.denoising_model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]

                batch = batch.to(self.device)
                b_size = batch.size(0)

                timesteps = torch.randint(1, self.T + 1, (b_size,), device=self.device)
                x_ts, epsilons = self.noising_model.sample_at_step(
                    batch=batch, t=timesteps
                )

                # Predict and compute MSE
                pred_epsilons = self.denoising_model(x_ts, timesteps)
                loss = F.mse_loss(pred_epsilons, epsilons)

                total_val_loss += loss.cpu().item()

        # switch back to training mode
        self.denoising_model.train()

        return total_val_loss / len(self.val_loader)

    def train(self, epochs):
        self.denoising_model.train()
        avg_train_loss = 0.0
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{epochs}")

            for batch in pbar:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]

                loss = self.train_step(batch)
                epoch_loss += loss

                pbar.set_postfix({"batch_loss": f"{loss:.4f}"})

            avg_train_loss = epoch_loss / len(self.train_loader)

            if self.val_loader is not None:
                val_loss = self.validate()
                print(
                    f"End of Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}"
                )
            else:
                print(f"End of Epoch {epoch} | Train Loss: {avg_train_loss:.4f}")

        print(f"Saving model with last avg_train_loss: {avg_train_loss:.4f}")

        run_path = MODEL_PATH / f"denoising-{avg_train_loss:.2f}-{epochs}.pt"
        torch.save(self.denoising_model.state_dict(), run_path)

        # Stable name so sampling has a default checkpoint to pick up
        torch.save(self.denoising_model.state_dict(), LATEST_CHECKPOINT)

        print(f"Saved {run_path} and {LATEST_CHECKPOINT}")
        return run_path
