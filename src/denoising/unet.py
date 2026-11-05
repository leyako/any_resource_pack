import torch
import torch.nn as nn
import torch.nn.functional as F

from group_norm import group_norm
from pos_embedding import sinusoidal_positional_embeddings


class AttentionGate(nn.Module):
    """
    This self attention between a gated signal (the upsampled x) and the skip connection
    allows us to keep only the interesting parts of the skip connection based on the incoming
    upsampled signal.

    First, we have to project both signals to the same number of channels (features) F_int,
    we do a 1x1 convolution to add new features to both g and x. We then combine the intermediate
    x and g and activate it with ReLU, we then do another projection from F_int features to only
    one, this allows for each pixel to have a single value between 0 and 1 and then with sigmoid
    we can know which pixels are the most important to keep and we weight it with that.
    """

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
    """
    Factorization of the 2 convolutions performed at each feature map

    Since it's for DDPM, we use the time embedding to give our model
    time-position-awareness for the given input image.

    The operation has a time_proj Linear layer to project the time embedding
    of dimension 512 to c_out dimensions that the first convolution gave as output.

    See https://apxml.com/courses/intro-diffusion-models/chapter-4-model-architecture-training/integrating-timestep-information
    for more information about time embedding integrations.
    """

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
    """
    Down class allows us to factorize the downsampling operation of our network.

    It uses time embeddings, directly in the DoubleConv layer.
    """

    def __init__(self, c_in, c_out, time_emb_dim):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(c_in, c_out, time_emb_dim)

    def forward(self, x, t_emb):
        x = self.pool(x)
        return self.conv(x, t_emb)


class Up(nn.Module):
    """
    Up class allows us to factorize the upsampling operation inside our network.

    It uses Attention gates to keep only interesting parts of the skip connection.

    It uses time embeddings during the convolutional blocks to teach the model time representations.
    """

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
    """
    This is the MLP that allows us to learn effective representations of time embeddings
    specially for our model's needs. This transforms a static fixed time embeddings to
    a learned one.
    """

    def __init__(self, base, time_emb_dim):
        super().__init__(
            nn.Linear(base, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )


class DenoisingUNet(nn.Module):
    """
    This is the main architecture for the Denoising model.

    base is the number of channels the designated layer will have during the convolutions.
    It grows from 64 to 1024, while reducing the dimension of our image from 32x32 to 4x4.

    We use the time embeddings that are here to help the model understand specific timesteps
    for the diffusion process. We use the sinusoidal_positional_embeddings from
    'Attention is All You Need' paper.

    To adapt the time embeddings to our model's needs, we use a TimeMLP which is a Linear, SiLU, Linear
    that transforms the static time embeddings into learned feature representations.

    To keep only the important parts of the skip connection, we perform use Attention Gates (AG).
    """

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
