# any_resource_pack

A DDPM (Denoising Diffusion Probabilistic Model) that generates 32x32 Minecraft
item textures, trained on sprites scraped from community PVP texture packs.

Thomas Trahant, Felix Muhlke, Yako Lemeilleur

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# The dataset lives in a git submodule hosted on Hugging Face
git submodule update --init --recursive
```

All runnable commands below are launched from the repository root.

## Configuration

Every hyperparameter is centralized in [`hyperparams/config.py`](hyperparams/config.py):
dataset paths, image size, batch size, number of timesteps `T`, the beta range,
the UNet widths, the learning rate, and the output folders.

The per-module `config.py` files (`src/training/`, `src/noising/`, `src/datasets/`)
only re-export from it, so one edit propagates everywhere.

Every CLI flag below defaults to the value in that file — flags are for
one-off experiments, the config file is for the project's real settings.

The device (`cuda`, `mps`, or `cpu`) is auto-detected; override with `--device`.

## Training

```bash
# Train with the settings from hyperparams/config.py
python -m src.training.train

# Override anything for an experiment
python -m src.training.train --epochs 50 --scheduler cosine --lr 1e-4 --batch-size 64
```

Checkpoints are written to `src/training/models/`:
- `denoising-<loss>-<epochs>.pt` — one file per run
- `denoising-latest.pt` — overwritten each run, used as the default for sampling

Useful flags: `--timesteps`, `--optim {adam,adamw}`, `--val-split`, `--rgba`
(train on 4-channel colour instead of the default grayscale+alpha), `--seed`.

## Generation

```bash
# Sample 32 images from the latest checkpoint
python -m src.sampling.sample

# From a specific checkpoint, into a specific file
python -m src.sampling.sample --checkpoint src/training/models/denoising-0.05-20.pt \
    --n 64 --out figures/final_samples.png --seed 0
```

Output goes to `figures/samples_<n>.png` by default. Add `--show` to also open a window.

Sampling runs the full reverse chain, so it costs `T` forward passes of the UNet
(1000 by default) — a few minutes on CPU for a batch of 32.

## Noising (forward process)

Visualizes `q(x_t | x_0)` at chosen timesteps — the figure used in the report's
noising section.

```bash
# Default: a strip at t = 0, 100, 250, 500, 750, 1000 on a training image
python -m src.noising.noise

# A single timestep, on your own sprite
python -m src.noising.noise --image path/to/sprite.png --t 500

# Compare schedules
python -m src.noising.noise --scheduler cosine --out figures/noising_cosine.png
```

## Report figures

Every figure in the report is regenerable from one command:

```bash
python make_figures.py
```

| Name | File | Needs |
|---|---|---|
| `schedules` | `figures/schedules.png` | nothing |
| `time-embedding` | `figures/time_embedding.png` | nothing |
| `noising` | `figures/noising_process.png` | the dataset |
| `grayscale` | `figures/grayscale_pipeline.png` | the dataset |
| `samples` | `figures/samples_32.png` | a trained checkpoint |

Figures are independent: any that cannot be built (dataset or checkpoint missing)
is reported and skipped, the rest still run. Build a subset with
`python make_figures.py --only schedules noising`.

The UNet architecture diagram is drawn in Typst inside the report itself, not here.

## Building the dataset

Only needed to rebuild the dataset from scratch — normally the submodule is enough.
Requires `HF_AUTH_TOKEN` in a `.env` file at the repository root.

```bash
python -m dataset.main
```

It scrapes pack manifests from `api.texturepack.be`, downloads the 42 item
sprites listed in `dataset/config.py`, removes duplicate textures, and pushes
the result to Hugging Face.

## Layout

```
hyperparams/config.py     every tunable value
make_figures.py           regenerates all report figures
src/datasets/             loading, RGBA->grayscale transforms, display helpers
src/noising/              forward process: variance schedules, NoisingModel, noise CLI
src/denoising/            the DenoisingUNet, attention gates, timestep embeddings
src/training/             trainer loop and train CLI
src/sampling/             reverse process: Sampler and sample CLI
dataset/                  scraping and dataset-building pipeline (run rarely)
figures/                  generated output, safe to delete
```
