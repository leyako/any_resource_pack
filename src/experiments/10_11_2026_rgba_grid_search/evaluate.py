import os
import csv
import gc
import copy
import time
import math
import traceback
from pathlib import Path

import torch
from torchvision.utils import save_image
from torchmetrics.image.fid import FrechetInceptionDistance

from src.datasets.dataset_loader import load_dataset
from src.datasets.utils import denormalize
from src.denoising.unet import DenoisingUNet
from src.noising.noising_model import NoisingModel
from src.sampling.sampler import Sampler
from src.experiments.config import CHUNK_BATCH_SIZE, TOTAL_REAL_IMAGES


def denormalize_and_mask(tensor: torch.Tensor, grayscale: bool = False) -> torch.Tensor:
    """
    1. Denormalizes from [-1, 1] to [0, 1]
    2. Uses alpha to mask the background to pure black
    """

    # denormalize
    denom = denormalize(tensor, grayscale=grayscale)
    channels = denom.size(1)
    if channels == 3:
        return denom

    rgb = denom[:, 0:3, :, :]
    alpha = denom[:, 3:4, :, :]

    # mask with alpha
    return rgb * alpha


def evaluate():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = CHUNK_BATCH_SIZE
    TOTAL_IMAGES = TOTAL_REAL_IMAGES  # Same as test set
    GRAYSCALE = False

    # setup Directories & CSV
    base_dir = Path(__file__).parent
    models_dir = base_dir / "models"
    grids_dir = base_dir / "grids"
    grids_dir.mkdir(parents=True, exist_ok=True)

    csv_path = base_dir / "evaluation_results.csv"
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "Model_Slug",
                    "Epochs",
                    "LR",
                    "Timesteps",
                    "Optimizer",
                    "FID_Score",
                    "Eval_Time_Seconds",
                ]
            )

    # initialize instance of FID Metric
    print("Initializing InceptionV3 and computing real statistics...")
    # normalize=True allows us to feed [0,1] float tensors directly
    base_fid = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)

    _, _, test_loader = load_dataset(grayscale=GRAYSCALE, val_split=0.2)
    channels = next(iter(test_loader))[0].size(1)
    # Compute real image stats exactly ONCE
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            batch = batch.to(DEVICE)
            prepared_real = denormalize_and_mask(batch, grayscale=GRAYSCALE)
            base_fid.update(prepared_real, real=True)

    print("Real statistics cached. Starting model evaluations...\n")

    # iterate over all saved models
    model_files = list(models_dir.glob("*.pt"))

    for idx, model_path in enumerate(model_files):
        slug = model_path.stem
        print(f"[{idx+1}/{len(model_files)}] Evaluating {slug}...")
        start_time = time.time()

        try:
            # load metadata and weights
            checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
            config = checkpoint["hyperparameters"]

            model = DenoisingUNet(
                in_channels=channels,
                out_channels=channels,
                T=config["T"],
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(DEVICE).eval()

            noising_model = NoisingModel(T=config["T"], device=DEVICE)

            sampler = Sampler(
                model=model,
                noising_model=noising_model,
                img_size=32,
                channels=channels,
                device=DEVICE,
            )

            with torch.no_grad():
                # generate Visual Grid
                grid_tensor = sampler.generate(batch_size=32)
                grid_rgb = denormalize_and_mask(grid_tensor, grayscale=GRAYSCALE)

                # torchvision's save_image is much cleaner than matplotlib for headless scripts
                save_image(grid_rgb, grids_dir / f"{slug}_grid.png", nrow=8)

                # FID Evaluation Loop using Deepcopy
                cloned_fid = copy.deepcopy(base_fid)

                generated_count = 0
                while generated_count < TOTAL_IMAGES:
                    # calculate how many to generate this step
                    current_batch = min(BATCH_SIZE, TOTAL_IMAGES - generated_count)

                    fake_tensor = sampler.generate(batch_size=current_batch)
                    fake_rgb = denormalize_and_mask(fake_tensor, grayscale=GRAYSCALE)

                    # update metric and immediately discard tensors to save VRAM
                    cloned_fid.update(fake_rgb, real=False)

                    generated_count += current_batch
                    del fake_tensor
                    del fake_rgb
                    torch.cuda.empty_cache()

                # Compute final FID
                fid_score = cloned_fid.compute().item()

            eval_time = time.time() - start_time
            print(f"  -> FID Score: {fid_score:.4f} | Time: {eval_time:.1f}s")

            # Log to CSV
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        slug,
                        config["epochs"],
                        config["lr"],
                        config["T"],
                        config["optim"],
                        f"{fid_score:.4f}",
                        f"{eval_time:.2f}",
                    ]
                )

        except Exception as e:
            error_msg = traceback.format_exc()
            print(f"  -> FAILED processing {slug}:\n{error_msg}")

        finally:
            # Aggressive VRAM Cleanup
            if "model" in locals():
                del model
            if "noising_model" in locals():
                del noising_model
            if "sampler" in locals():
                del sampler
            if "cloned_fid" in locals():
                del cloned_fid

            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    evaluate()
