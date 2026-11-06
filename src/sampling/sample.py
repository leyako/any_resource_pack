from src.datasets.dataset_loader import load_dataset
from src.denoising.unet import DenoisingUNet
from src.noising.noising_model import NoisingModel
from src.training.config import MODEL_PATH
from src.sampling.sampler import Sampler

from src.datasets.utils import show_image

import torch


def sample():
    device = "cuda"

    train_loader, _, _ = load_dataset(grayscale=True)
    channels = next(iter(train_loader))[0].size(1)

    model = DenoisingUNet(in_channels=channels, out_channels=channels, T=1000)
    model.load_state_dict(
        torch.load(MODEL_PATH / "denoising_0.0545.pt", map_location=device)
    )
    model.to(device).eval()

    noising_model = NoisingModel(T=1000, scheduler="linear", device=device)

    sampler = Sampler(
        model=model,
        noising_model=noising_model,
        img_size=32,
        channels=channels,
        device=device,
    )

    with torch.no_grad():
        generated_images = sampler.generate(batch_size=32)

    print(f"Generated tensor shape: {generated_images.shape}")

    show_image(generated_images, grayscale=True)


if __name__ == "__main__":
    sample()
