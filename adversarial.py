from pathlib import Path
from datetime import datetime

import torch 
import torch.nn as nn
import numpy as np
import typer

from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from PIL import Image
from torch.optim.lr_scheduler import OneCycleLR
from dataset.real_images import RealImages
from training_clipping import load_progan

app = typer.Typer()

torch.manual_seed(0)
np.random.seed(0)


def find_noise_mse(writer, frame, progan, img, lr, num_iters=300):
    # noise = torch.randn(img.shape[0], 512)
    noise, _ = progan.buildNoiseData(img.shape[0], torch.tensor([[1]] * img.shape[0]).long())
    
    noise.requires_grad = True
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam([noise], lr=lr)
    scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=num_iters)
    metrics = []
    progress = tqdm(range(num_iters))
    for e in progress:
        output = progan.netG(noise)
        cost = loss(output, img)
        writer.add_scalar(f'Loss/MSE loss/frame_{frame}', cost.item(), e)
        metrics.append(cost.item())

        optimizer.zero_grad()
        cost.backward()
        # freeze one hot
        noise.grad[:, 512:] = 0
        optimizer.step()
        scheduler.step()
        progress.set_description(f"loss {cost.item()}")

    return output, cost.item(), noise


def from_torch(img):
    npimg = np.moveaxis(img.detach().cpu().numpy(), 0, 2)
    img = (npimg - npimg.min()) / (npimg.max() - npimg.min())
    return img


@app.command()
def find_noise_for_images(
    model_path: Path = Path("/shared/results/struski/videoGAN/AC-ProGAN_2022-10-14_141007/jelita"), 
    model_name: str = "jelita", 
    images_path: Path = Path("/shared/results/Skopia/images_split2classes/polip/"),
    output_dir: Path = Path("IMAGES"), 
    lr: float = 1.0, 
    num_iters: int = 300,
):
    now = datetime.now()
    date = now.strftime("%Y.%m.%d_%H.%M.%S")
    name = f'projections_24/baseline_{date}'
    log_writer = SummaryWriter(f'runs/{name}')
    
    progan = load_progan(model_name, str(model_path))
    kwargs = {"num_workers": 8, "pin_memory": True, "batch_size": 24, "shuffle": False, "drop_last": False}
    dataloader = DataLoader(RealImages(root=images_path), **kwargs)
    progan.netD.eval()
    image_output_dir = output_dir / "images"
    tensor_output_dir = output_dir / "noise"
    image_output_dir.mkdir(parents=True, exist_ok=True)
    tensor_output_dir.mkdir(parents=True, exist_ok=True)
    output_dir

    j = 0
    for e, (real_image, paths) in enumerate(dataloader):
        real_image = real_image.cuda()
        output, cost, noise = find_noise_mse(log_writer, e, progan, real_image, lr=lr, num_iters=num_iters)
        noise = noise.detach().cpu()

        for img1, img2, path, single_noise in zip(real_image, output, paths, noise):
            i1 = (from_torch(img1) * 255).astype(np.uint8)
            i2 = (from_torch(img2) * 255).astype(np.uint8)
            img = np.column_stack([i1, i2])
            name = Path(path).stem
            Image.fromarray(img).save(f"{image_output_dir}/{name}.jpeg")
            torch.save(single_noise, f"{tensor_output_dir}/{name}.pt")
            j += 1


if __name__ == "__main__":
    app()