import torch
import math


def sinusoidal_positional_embeddings(timesteps: int = 1000, d_model: int = 512):
    """
    Computes sinusoidal timestep positional embeddings

    Returns a (timesteps, d_model) tensor used to encode the diffusion timestep 't'.
    This gives the model a sense of the current noise level by interleaving sine and
    cosine waves of varying frequencies across the embedding dimensions.

    The implementation uses a math trick: it computes the denominator using a log-space
    equivalent (torch.exp) rather than the raw paper formula (1 / 10000^(2i/d_model)). This
    avoids expensive hardware power calculations and prevents floating-point
    precision issues (like underflow) when working with large numbers.
    (These researchers are so smart...)
    """
    position = torch.arange(0, timesteps, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )  # This is the log-space equivalent

    embeddings = torch.zeros(timesteps, d_model)
    embeddings[:, 0::2] = torch.sin(position * div_term)
    embeddings[:, 1::2] = torch.cos(position * div_term)
    return embeddings
