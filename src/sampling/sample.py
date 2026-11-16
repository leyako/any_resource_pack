import argparse
from pathlib import Path

import torch

from hyperparams.config import (
    FIGURES_PATH,
    IMG_SIZE,
    RGBA_MODEL,
    GRAYSCALE_MODEL,
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
        "--model",
        default="rgba",
        help="'rgba' or 'grayscale'.",
    )
    parser.add_argument("--n", type=int, default=32, help="How many images to sample.")
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
    model_path = RGBA_MODEL if args.model == "rgba" else GRAYSCALE_MODEL
    run_slug = model_path.stem

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    hp = checkpoint.get("hyperparameters", {})

    timesteps = hp.get("T") or hp.get("timesteps") or TIMESTEPS
    img_size = hp.get("img_size", IMG_SIZE)
    base = hp.get("base", 64)
    use_skip = hp.get("use_skip", True)
    channels = checkpoint["model_state_dict"]["enc1.conv1.0.weight"].shape[1]

    grayscale = channels == 2

    print(
        f"Device: {device} | checkpoint: {run_slug} | "
        f"T={timesteps} | channels={channels} | img_size={img_size}"
    )

    model = DenoisingUNet(
        in_channels=channels,
        out_channels=channels,
        base=base,
        use_skip=use_skip,
        T=timesteps,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    noising_model = NoisingModel(T=timesteps, device=device)

    sampler = Sampler(
        model=model,
        noising_model=noising_model,
        img_size=img_size,
        channels=channels,
        device=device,
    )

    with torch.no_grad():
        generated_images = sampler.generate(batch_size=args.n)

    print(f"Generated tensor shape: {generated_images.shape}")

    out_path = args.out or FIGURES_PATH / f"samples_{args.n}.png"
    show_image(
        generated_images,
        slug=run_slug,
        grayscale=grayscale,
        out_path=out_path,
        show=args.show,
    )


if __name__ == "__main__":
    sample()
