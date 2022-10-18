from pathlib import Path

import torch 
import torch.nn as nn
import numpy as np
import typer

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from PIL import Image
from torch.optim.lr_scheduler import OneCycleLR
from dataset.real_images import RealImages
from training_clipping import load_progan, RealVideos
from monai.data import list_data_collate

app = typer.Typer()


def find_noise_mse(progan, img, lr, num_iters=300):
    noise = torch.randn(img.shape[0], 512)
    noise.requires_grad = True
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam([noise], lr=lr)
    scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=num_iters)
    metrics = []
    progress = tqdm(range(num_iters))
    for _ in progress:
        output = progan.netG(noise)
        cost = loss(output, img)
        metrics.append(cost.item())

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        scheduler.step()
        progress.set_description(f"loss {cost.item()}")

    return output, cost.item()


def from_torch(img):
    npimg = np.moveaxis(img.detach().cpu().numpy(), 0, 2)
    img = (npimg - npimg.min()) / (npimg.max() - npimg.min())
    return img


@app.command()
def find_noise_for_images(
    model_path: Path = Path("/shared/results/z1143165/polipy"), 
    model_name: str = "polipy", 
    images_path: Path = Path("/shared/results/Skopia/videos24frames"),
    output_dir: Path = Path("IMAGES"), 
    n_batches: int = 20, 
    lr: float = 1.0, 
    num_iters: int = 300,
):
    progan = load_progan(model_name, str(model_path))
    kwargs = {"num_workers": 8, "pin_memory": True, "batch_size": 24, "shuffle": False, "drop_last": False}
    dataloader = DataLoader(RealImages(root=images_path), **kwargs)
    progan.netD.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    j = 0
    costs = []
    for e, real_image in enumerate(dataloader):
        if e == n_batches:
            break
        real_image = real_image.cuda()
        output, cost = find_noise_mse(progan, real_image, lr=lr, num_iters=num_iters)
        costs.append(cost)

        for img1, img2 in zip(real_image, output):
            i1 = (from_torch(img1) * 255).astype(np.uint8)
            i2 = (from_torch(img2) * 255).astype(np.uint8)
            img = np.column_stack([i1, i2])
            Image.fromarray(img).save(f"{output_dir}/{j}.jpeg")
            j += 1

    print(np.mean(costs))
    return np.mean(costs)


if __name__ == "__main__":
    app()