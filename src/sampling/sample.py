import argparse

import torch

from hyperparams.config import (
    FIGURES_PATH,
    IMG_SIZE,
    LATEST_CHECKPOINT,
    SCHEDULER,
    TIMESTEPS,
    get_device,
    set_seed,
)
from src.datasets.utils import show_image
from src.denoising.unet import DenoisingUNet
from src.noising.noising_model import NoisingModel
from src.sampling.sampler import Sampler


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate texture samples from a trained DDPM checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        default=str(LATEST_CHECKPOINT),
        help="Path to the model weights (.pt).",
    )
    parser.add_argument("--n", type=int, default=32, help="How many images to sample.")
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS, help="T")
    parser.add_argument("--scheduler", choices=["linear", "cosine"], default=SCHEDULER)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument(
        "--rgba",
        action="store_true",
        help="Model was trained on 4-channel RGBA instead of grayscale+alpha.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG (default: figures/samples_<n>.png).",
    )
    parser.add_argument(
        "--device", default=None, help="cuda | mps | cpu (auto-detected by default)"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--show", action="store_true", help="Open a window as well as saving."
    )
    return parser.parse_args(argv)


def sample(argv=None):
    args = parse_args(argv)

    if args.seed is not None:
        set_seed(args.seed)

    device = get_device(args.device)
    grayscale = not args.rgba
    # grayscale pipeline keeps luminance + alpha, RGBA keeps all four
    channels = 2 if grayscale else 4

    print(f"Device: {device} | checkpoint: {args.checkpoint}")

    model = DenoisingUNet(in_channels=channels, out_channels=channels, T=args.timesteps)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    noising_model = NoisingModel(
        T=args.timesteps, scheduler=args.scheduler, device=device
    )

    sampler = Sampler(
        model=model,
        noising_model=noising_model,
        img_size=args.img_size,
        channels=channels,
        device=device,
    )

    with torch.no_grad():
        generated_images = sampler.generate(batch_size=args.n)

    print(f"Generated tensor shape: {generated_images.shape}")

    out_path = args.out or FIGURES_PATH / f"samples_{args.n}.png"
    show_image(generated_images, grayscale=grayscale, out_path=out_path, show=args.show)


if __name__ == "__main__":
    sample()
