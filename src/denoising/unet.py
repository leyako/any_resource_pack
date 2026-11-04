import torch
import torch.nn as nn
import torch.nn.functional as F

from group_norm import group_norm
from pos_embedding import sinusoidal_positional_embeddings


class AttentionGate(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()

        # 1x1 conv to project the gating signal (the incoming upsample) 'F_int' channels
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1)

        # 1x1 conv to project the skip connection 'x' to 'F_int' channels
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1)

        # 1x1 conv to project the combined signal from 'F_int' down to 1 channel
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        g_proj = self.W_g(g)
        x_proj = self.W_x(x)

        add_intermediate = g_proj + x_proj

        sigma_1 = self.relu(add_intermediate)

        psi_proj = self.psi(sigma_1)

        sigma_2 = self.sigmoid(psi_proj)
        # F.interpolate(sigma_2, x.shape[2:], mode="bilinear") # If we want memory efficiency, W_g should have stride=2 too.
        x_hat = sigma_2 * x

        return x_hat


class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out, time_emb_dim=512):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),  # DONE
            nn.BatchNorm2d(
                c_out
            ),  # à remplacer par la group norm mais je sais pas l'idéal
            nn.ReLU(),
        )

        self.time_proj = nn.Linear(time_emb_dim, c_out)

        self.conv2 = nn.Sequential(
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.BatchNorm2d(
                c_out
            ),  # à remplacer par la group norm mais je sais pas l'idéal
            nn.ReLU(),
        )

    def forward(self, x, t_emb):
        x = self.conv1(x)

        t_emb_proj = self.time_proj(t_emb).reshape(
            x.shape[0], x.shape[1], 1, 1
        )  # Reshape t_emb to match dim number of x

        x = x + t_emb_proj
        return self.conv2(x)


class Down(nn.Module):
    def __init__(self, c_in, c_out, time_emb_dim):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(c_in, c_out, time_emb_dim)

    def forward(self, x, t_emb):
        x = self.pool(x)
        return self.conv(x, t_emb)


class Up(nn.Module):
    def __init__(self, c_in: int, c_out: int, time_emb_dim: int, use_skip=True):
        super().__init__()
        self.use_skip = use_skip
        self.up = nn.ConvTranspose2d(c_in, c_in // 2, kernel_size=2, stride=2)

        if self.use_skip:
            self.attngate = AttentionGate(c_in // 2, c_in // 2, c_in // 4)

        self.conv = DoubleConv(
            c_in if self.use_skip else c_in // 2, c_out, time_emb_dim
        )

    def forward(self, x, skip, t_emb):
        g = self.up(x)
        if self.use_skip:
            x_hat = self.attngate(g, skip)
            x = torch.cat([g, x_hat], dim=1)
        else:
            x = g

        x = self.conv(x, t_emb)
        return x


class TimeMLP(nn.Sequential):
    def __init__(self, base, time_emb_dim):
        super().__init__(
            nn.Linear(base, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )


class DenoisingUNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        base: int = 64,
        use_skip: bool = True,
        T: int = 1000,
    ) -> None:
        super().__init__()
        c = [
            base,  # 64
            base * 2,  # 128
            base * 4,  # 256
            base * 8,  # 512
            base * 16,  # 1024
        ]  # Number of channels per feature map

        time_emb_dim = 512
        time_embeddings = sinusoidal_positional_embeddings(T, time_emb_dim)
        self.register_buffer("time_embeddings", time_embeddings)

        self.time_mlp = TimeMLP(time_emb_dim, time_emb_dim)

        # First feature map
        self.enc1 = DoubleConv(in_channels, c[0])

        # Second feature map
        self.enc2 = Down(c[0], c[1], time_emb_dim)

        # Third feature map
        self.enc3 = Down(c[1], c[2], time_emb_dim)

        # Fourth feature map
        self.enc4 = Down(c[2], c[3], time_emb_dim)

        # The bottleneck at 1024 channel features
        self.bottleneck = Down(c[3], c[4], time_emb_dim)

        # Upsamples with attention gates
        self.up4 = Up(c[4], c[3], time_emb_dim, use_skip)
        self.up3 = Up(c[3], c[2], time_emb_dim, use_skip)
        self.up2 = Up(c[2], c[1], time_emb_dim, use_skip)
        self.up1 = Up(c[1], c[0], time_emb_dim, use_skip)

        # Out
        self.out = nn.Conv2d(c[0], out_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_embeddings[t]
        t_emb = self.time_mlp(t_emb)

        enc1 = self.enc1(x, t_emb)
        enc2 = self.enc2(enc1, t_emb)
        enc3 = self.enc3(enc2, t_emb)
        enc4 = self.enc4(enc3, t_emb)
        x = self.bottleneck(enc4, t_emb)
        x = self.up4(x, enc4, t_emb)
        x = self.up3(x, enc3, t_emb)
        x = self.up2(x, enc2, t_emb)
        x = self.up1(x, enc1, t_emb)

        return self.out(x)
