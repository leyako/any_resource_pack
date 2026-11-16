from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from tqdm import tqdm

from hyperparams.config import (
    BASE_CHANNELS,
    LEARNING_RATE,
    MODEL_PATH,
    SCHEDULER,
)
from src.denoising.unet import DenoisingUNet
from src.noising.noising_model import NoisingModel


class DenoisingTrainer:
    def __init__(
        self,
        train_loader,
        val_loader,
        optim: Literal["adam", "adamw"],
        T: int,
        device: str = "cuda",
        lr: float = LEARNING_RATE,
        base: int = BASE_CHANNELS,
        scheduler: str = SCHEDULER,
    ) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.T = T
        self.device = device
        self.lr = lr
        self.optim = optim
        self.scheduler = scheduler

        batch, _ = next(iter(train_loader))
        self.denoising_model = DenoisingUNet(
            batch.size(1), batch.size(1), base=base, T=T
        ).to(self.device)

        if optim == "adam":
            self.optimizer = torch.optim.Adam(self.denoising_model.parameters(), lr=lr)
        else:
            self.optimizer = torch.optim.AdamW(self.denoising_model.parameters(), lr=lr)

        self.noising_model = NoisingModel(T=T, device=self.device)

        self.model_slug = f"{optim}-lr{lr}-T{T}"

    def train_step(self, batch):
        batch = batch.to(self.device)
        b_size = batch.size(0)

        # Sample random timesteps
        timesteps = torch.randint(1, self.T + 1, (b_size,), device=self.device)
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

    def save_epoch(self, config, save_dir, val_loss, train_loss):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.denoising_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metadata": {
                "final_val_loss": val_loss,
                "train_loss": train_loss,
            },
            "hyperparameters": config,
        }

        run_slug = (
            f"run_lr{config['lr']}_T{config['T']}_"
            f"{config['optim']}_{config.get('scheduler', 'linear')}_ep{config['epochs']}"
        )
        save_path = save_dir / f"{run_slug}.pt"
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint: {save_path}")

    def train(self, epochs, save_epochs=None, config=None, save_dir=None):
        self.denoising_model.train()
        avg_train_loss = 0.0
        val_loss = 0.0

        base_config = {
            "lr": self.lr,
            "T": self.T,
            "optim": self.optim,
            "epochs": epochs,
            "scheduler": self.scheduler,
        }
        if config is not None:
            base_config.update(config)

        target_dir = Path(save_dir) if save_dir is not None else Path(MODEL_PATH)

        epochs_to_save = set(save_epochs) if save_epochs is not None else set()
        epochs_to_save.add(epochs)

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

            if epoch in epochs_to_save:
                current_config = base_config.copy()
                current_config["epochs"] = epoch
                self.save_epoch(current_config, target_dir, val_loss, avg_train_loss)

        return val_loss
