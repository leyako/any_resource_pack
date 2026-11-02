from src.datasets.dataset_loader import load_dataset
from src.noising.noising_model import NoisingModel
import matplotlib.pyplot as plt
from src.datasets.utils import denormalize

train_loader, test_loader = load_dataset(grayscale=False)
images, _ = next(iter(train_loader))

x = images[0]

noising_model = NoisingModel(T=1000, x_0=x, scheduler="linear")

new_x = noising_model.sample_at_step(250)

new_x = denormalize(new_x, grayscale=False)

new_x = new_x.permute(1, 2, 0).cpu().numpy()


plt.imshow(new_x)
plt.savefig("noised_example.png", bbox_inches="tight", dpi=150)
