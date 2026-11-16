"""
Regenerates every figure of the report into figures/.

    python3 make_figures.py              # all figures that can be built
    python3 make_figures.py --only schedules noising

No figure is edited by hand: each is rebuilt from the code and the config, so a
hyperparameter change propagates to the report in one command. Figures are
independent, and one that cannot be built (no dataset, no checkpoint) is skipped.
"""

import argparse
import traceback

import matplotlib.pyplot as plt

from hyperparams.config import (
    BETA_END,
    BETA_START,
    FIGURES_PATH,
    LATEST_CHECKPOINT,
    TIMESTEPS,
    set_seed,
)

SEED = 0


def fig_schedules():
    """
    beta_t and alpha_bar_t under both schedules. Needs no dataset.

    alpha_bar_t is the readable one: the fraction of signal left after t steps.
    """
    from src.noising.variance_scheduling import cosine_scheduler, linear_scheduler

    lin_betas, _, lin_bar = linear_scheduler(T=TIMESTEPS, start=BETA_START, end=BETA_END)
    cos_betas, _, cos_bar = cosine_scheduler(T=TIMESTEPS, start=BETA_START, end=BETA_END)

    _, axes = plt.subplots(1, 2, figsize=(10, 3.6))

    axes[0].plot(lin_betas.numpy(), label="linear (sqrt interp.)")
    axes[0].plot(cos_betas.numpy(), label="cosine")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel(r"$\beta_t$")
    axes[0].set_title("Variance schedule")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(lin_bar.numpy(), label="linear (sqrt interp.)")
    axes[1].plot(cos_bar.numpy(), label="cosine")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel(r"$\bar{\alpha}_t$")
    axes[1].set_title("Signal retained after t steps")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = FIGURES_PATH / "schedules.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved figure to {out}")


def fig_time_embedding():
    """
    Heatmap of the sinusoidal timestep embeddings. Needs no dataset.

    Low dimensions oscillate fast (adjacent timesteps), high ones slowly (early
    vs late diffusion).
    """
    from src.denoising.pos_embedding import sinusoidal_positional_embeddings

    emb = sinusoidal_positional_embeddings(TIMESTEPS, 512)

    plt.figure(figsize=(7, 3.4))
    plt.imshow(emb.T.numpy(), aspect="auto", cmap="RdBu", origin="lower")
    plt.colorbar(label="value")
    plt.xlabel("timestep t")
    plt.ylabel("embedding dimension")
    plt.title("Sinusoidal timestep embeddings")

    plt.tight_layout()
    out = FIGURES_PATH / "time_embedding.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved figure to {out}")


def fig_noising():
    """The forward process on 3 random dataset sprites. Needs the dataset."""
    from src.noising.noise import main as noise_main

    noise_main()


def fig_grayscale():
    """Raw RGBA sprite next to its luminance+alpha reduction. Needs the dataset."""
    from src.datasets.test import show_grayscale_example

    show_grayscale_example(show=False, verbose=False)


def fig_samples():
    """Grid of generated textures. Needs a checkpoint. Seeded, so reproducible."""
    if not LATEST_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"No checkpoint at {LATEST_CHECKPOINT}. Train first: "
            "`python -m src.training.train`"
        )

    from src.sampling.sample import sample

    sample(["--n", "32", "--seed", str(SEED)])


FIGURES = {
    "schedules": fig_schedules,
    "time-embedding": fig_time_embedding,
    "noising": fig_noising,
    "grayscale": fig_grayscale,
    "samples": fig_samples,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(FIGURES),
        default=sorted(FIGURES),
        help="Build only these figures.",
    )
    args = parser.parse_args()

    set_seed(SEED)

    built, skipped = [], []
    for name in args.only:
        print(f"\n=== {name} ===")
        try:
            FIGURES[name]()
            built.append(name)
        except Exception as exc:
            skipped.append((name, exc))
            print(f"SKIPPED {name}: {exc}")
            traceback.print_exc()

    print(f"\nBuilt: {', '.join(built) if built else 'nothing'}")
    if skipped:
        print(f"Skipped: {', '.join(name for name, _ in skipped)}")


if __name__ == "__main__":
    main()
