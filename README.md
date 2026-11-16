# any_resource_pack

A DDPM (Denoising Diffusion Probabilistic Model) that generates 32x32 Minecraft
item textures, trained on sprites scraped from community PVP texture packs.

Thomas Trahant, Felix Muhlke, Yako Lemeilleur

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
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
python3 -m src.training.train

# Override anything for an experiment
python3 -m src.training.train --epochs 50 --scheduler cosine --lr 1e-4 --batch-size 64
```

Checkpoints are written to `src/training/models/`:
- `denoising-<loss>-<epochs>.pt` — one file per run
- `denoising-latest.pt` — overwritten each run, used as the default for sampling

Useful flags: `--timesteps`, `--optim {adam,adamw}`, `--val-split`, `--rgba`
(train on 4-channel colour instead of the default grayscale+alpha), `--seed`.

## Generation

```bash
# Sample 32 images from the latest checkpoint
python3 -m src.sampling.sample

# From a specific checkpoint, into a specific file
python3 -m src.sampling.sample --checkpoint src/training/models/denoising-0.05-20.pt \
    --n 64 --out figures/final_samples.png --seed 0
```

Output goes to `figures/samples_<n>.png` by default. Add `--show` to also open a window.

Sampling runs the full reverse chain, so it costs `T` forward passes of the UNet
(1000 by default) — a few minutes on CPU for a batch of 32.

## Noising (forward process)

Applies `q(x_t | x_0)` to 3 sprites drawn at random from the dataset, at 7
timesteps evenly spaced over `[0, T]`. This is the figure used in the report's
noising section.

```bash
python3 -m src.noising.noise
```

Output goes to `figures/noising_process.png`: one row per sprite, one column per
timestep. With `T = 1000` the timesteps are 0, 167, 333, 500, 667, 833 and 1000,
and the paths actually drawn are printed. Every run draws different sprites.

The command takes no arguments. To change what it produces, edit `N_IMAGES` and
`N_STEPS` in `src/noising/noise.py`, or `TIMESTEPS`, `SCHEDULER` and `GRAYSCALE`
in `hyperparams/config.py`.

It needs only torch and pillow, so it works without torchvision.

### Checking the forward process

```bash
python3 -m src.noising.check
```

Noises a synthetic sprite and asserts that the measured standard deviation matches
what the schedule predicts, that a fixed seed reproduces the same result, and that
`alpha_bar` reaches its expected extremes. Needs neither the dataset nor torchvision.

## Report figures

Every figure in the report is regenerable from one command:

```bash
python3 make_figures.py
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
`python3 make_figures.py --only schedules noising`.

The UNet architecture diagram is drawn in Typst inside the report itself, not here.

## Building the dataset

Only needed to rebuild the dataset from scratch — normally the submodule is enough.
Requires `HF_AUTH_TOKEN` in a `.env` file at the repository root.

```bash
python3 -m dataset.main
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
