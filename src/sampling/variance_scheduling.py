import torch


def reverse_var_scheduler(betas):
    return torch.sqrt(betas)
