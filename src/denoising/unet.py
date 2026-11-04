import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
