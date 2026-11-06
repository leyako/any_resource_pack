import torch

from .config import TRAIN_PATH, TEST_PATH, BATCH_SIZE
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import v2, Grayscale

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_cuda = device.type == "cuda"


class RGBAToGrayscale:
    """
    Converts RGBA Tensor [4, 32, 32] to Grayscale RGBA [2, 32, 32].
    Computes luminance across RGB.
    Alpha channel remains completely untouched.
    """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor is [4, 32, 32] in range [0.0, 1.0]
        rgb = tensor[:3, :, :]
        alpha = tensor[3:, :, :]

        # Calculate luminance (Y = 0.299 R + 0.587 G + 0.114 B)
        weights = torch.tensor([0.299, 0.587, 0.114], device=tensor.device).view(
            3, 1, 1
        )
        gray = (rgb * weights).sum(dim=0, keepdim=True)  # Shape: [1, 32, 32]

        # Recombine with Alpha -> Shape: [2, 32, 32]
        return torch.cat([gray, alpha], dim=0)


class RGBAColorJitter:
    """
    Applies ColorJitter strictly to RGB channels to protect Alpha transparency.
    """

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05):
        self.jitter = v2.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        rgb = tensor[:3, :, :]
        alpha = tensor[3:, :, :]
        rgb_jittered = self.jitter(rgb)
        return torch.cat([rgb_jittered, alpha], dim=0)


class LuminanceJitter:
    """
    Applies random brightness and contrast adjustments
    strictly to the Luminance channel [0] of a 2-channel [2, 32, 32] tensor.
    Leaves Alpha untouched.
    """

    def __init__(self, brightness: float = 0.15, contrast: float = 0.15):
        # Apply jitter on a 1-channel tensor
        self.jitter = v2.ColorJitter(brightness=brightness, contrast=contrast)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor is [2, 32, 32] in range [0.0, 1.0]
        lum = tensor[0:1, :, :]  # [1, 32, 32]
        alpha = tensor[1:2, :, :]  # [1, 32, 32]

        # Jitter brightness & contrast on luminance
        lum_jittered = self.jitter(lum)

        return torch.cat([lum_jittered, alpha], dim=0)


def get_transforms(grayscale: bool = False, is_train: bool = True):
    transform_list = []
    # Convert to RGBA to get the alpha value
    transform_list.append(
        v2.Lambda(lambda img: v2.functional.to_image(img.convert("RGBA")))
    )
    # Convert to float32 and scale values between 0 and 1
    transform_list.append(
        v2.Lambda(
            lambda img: v2.functional.to_dtype(img, dtype=torch.float32, scale=True)
        ),
    )

    if grayscale:
        # Convert RGB to Grayscale and keep only 2 channel (grayscale + luminance)
        transform_list.append(RGBAToGrayscale())

        if is_train:
            transform_list.append(LuminanceJitter(brightness=0.15, contrast=0.15))
    else:
        if is_train:
            # Apply Color Jitter (only during training, only on full color)
            transform_list.append(
                RGBAColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            )

    # Normalize 2 channels to [-1.0, 1.0] (according to the ddpm paper) with 0.5 param
    if grayscale:
        transform_list.append(v2.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]))
    else:
        transform_list.append(
            v2.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5])
        )

    return v2.Compose(transform_list)


def rgba_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        # Convert explicitly to RGBA
        return img.convert("RGBA")


def load_dataset(grayscale: bool = False, val_split: float = 0.2):
    train_tf = get_transforms(grayscale=grayscale, is_train=True)
    test_tf = get_transforms(grayscale=grayscale, is_train=False)

    base_train_ds = datasets.ImageFolder(
        TRAIN_PATH, transform=train_tf, loader=rgba_loader
    )
    base_val_ds = datasets.ImageFolder(
        TRAIN_PATH, transform=test_tf, loader=rgba_loader
    )

    # This is used for the val_ds' creation
    dataset_size = len(base_train_ds)
    indices = torch.randperm(dataset_size).tolist()

    val_size = int(val_split * dataset_size)

    val_ds = Subset(base_val_ds, indices[:val_size])
    train_ds = Subset(base_train_ds, indices[val_size:])

    test_ds = datasets.ImageFolder(TEST_PATH, transform=test_tf, loader=rgba_loader)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=is_cuda
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=is_cuda
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=is_cuda
    )

    return train_loader, val_loader, test_loader
