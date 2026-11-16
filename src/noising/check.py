"""
Self-check of the forward process. Needs neither the dataset nor torchvision.

    python3 -m src.noising.check

Noises a synthetic sprite at increasing timesteps and asserts that the measured
statistics match what the schedule predicts.
"""

import torch

from hyperparams.config import IMG_SIZE, SCHEDULER, TIMESTEPS, set_seed
from src.noising.noising_model import NoisingModel


def synthetic_batch(batch_size: int = 256, channels: int = 2) -> torch.Tensor:
    """A crude sprite in [-1, 1]: a bright square on a dark background."""
    x = -torch.ones(batch_size, channels, IMG_SIZE, IMG_SIZE)
    x[:, :, 8:24, 8:24] = 1.0
    return x


def main():
    set_seed(0)

    model = NoisingModel(T=TIMESTEPS, scheduler=SCHEDULER, device="cpu")
    x0 = synthetic_batch()

    print(f"schedule={SCHEDULER}  T={TIMESTEPS}  x0 std={x0.std():.3f}")
    print(f"{'t':>6} {'alpha_bar':>10} {'std(x_t)':>9} {'expected':>9} {'corr(x0)':>9}")

    for t in [1, 100, 250, 500, 750, TIMESTEPS]:
        t_tensor = torch.full((x0.size(0),), t, dtype=torch.long)
        x_t, eps = model.sample_at_step(batch=x0, t=t_tensor)

        alpha_bar = model.sigmas_bar[t - 1]

        # Var(x_t) = alpha_bar * Var(x0) + (1 - alpha_bar), the two terms being
        # independent, so the standard deviation follows.
        expected = torch.sqrt(alpha_bar * x0.var() + (1 - alpha_bar))

        # Correlation with x0 measures how much of the signal survives.
        corr = torch.corrcoef(torch.stack([x0.flatten(), x_t.flatten()]))[0, 1]

        print(
            f"{t:>6} {alpha_bar:>10.4f} {x_t.std():>9.3f} "
            f"{expected:>9.3f} {corr:>9.3f}"
        )

        assert x_t.shape == x0.shape, "noising must preserve shape"
        assert eps.shape == x0.shape, "epsilon must match the input shape"
        assert torch.isfinite(x_t).all(), "noised batch contains NaN or inf"
        assert abs(x_t.std() - expected) < 0.05, (
            f"std mismatch at t={t}: {x_t.std():.3f} vs {expected:.3f}"
        )

    # The closed form must be reproducible under a fixed seed.
    set_seed(1)
    a, _ = model.sample_at_step(batch=x0, t=torch.full((x0.size(0),), 500))
    set_seed(1)
    b, _ = model.sample_at_step(batch=x0, t=torch.full((x0.size(0),), 500))
    assert torch.equal(a, b), "same seed must give the same noised batch"

    # At t = T almost no signal is left; at t = 1 almost all of it is.
    assert model.sigmas_bar[-1] < 0.01, "x_T should be near-pure noise"
    assert model.sigmas_bar[0] > 0.99, "x_1 should be barely noised"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
