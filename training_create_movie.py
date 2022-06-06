import argparse
import json
from pathlib import Path
from random import randint

import cv2
import torch
from PIL import Image
from torchvision.utils import make_grid
from tqdm import tqdm

from models.frame_seed_generator import FrameSeedGenerator
from models.utils.utils import loadmodule, getLastCheckPoint, getNameAndPackage, parse_state_name
from visualization.visualizer import resizeTensor


def load_progan(name='jelito3d_batchsize8', checkPointDir='/checkpoints'):
    checkpointData = getLastCheckPoint(checkPointDir, name, scale=6, iter=96000)
    modelConfig, pathModel, tmpconfig = checkpointData
    _, scale, _ = parse_state_name(pathModel)

    module = 'PGAN'
    packageStr, modelTypeStr = getNameAndPackage(module)
    modelType = loadmodule(packageStr, modelTypeStr)

    with open(modelConfig, 'rb') as file:
        config = json.load(file)

    model = modelType(useGPU=True, storeAVG=True, **config)
    model.load(pathModel)

    model = model.avgG
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runner for VAE model")
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--n_frames", type=int, default=24)
    parser.add_argument("--size_img", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--resume_path", type=str, required=True)
    parser.add_argument("--name", type=str, default='')
    args = parser.parse_args()
    print(args)

    w_video = h_video = 256
    padding = 2

    seed_ = randint(10, 10000)
    torch.manual_seed(seed_)
    torch.cuda.manual_seed(seed_)

    save_results = f'{str(Path(args.resume_path).parents[1])}/{Path(__file__).resolve().stem.removeprefix("training_")}{args.name}'
    print(f'\033[0;33m{save_results}\033[0m')

    Path(save_results).mkdir(parents=True, exist_ok=True)
    time = torch.arange(args.n_frames).tile(args.batch_size).unsqueeze(1).cuda()

    proganG = load_progan()
    proganG.eval()

    fsg = FrameSeedGenerator().cuda()
    fsg.load_state_dict(torch.load(args.resume_path))
    fsg.eval()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    torch.cuda.empty_cache()
    with torch.no_grad():

        for i in tqdm(range(args.num_samples), desc='Samples'):
            noise = torch.randn([args.batch_size, 2047]).unsqueeze(1).tile(1, args.n_frames, 1).view(-1, 2047).cuda()
            latent = fsg(noise, time)
            fake_video = proganG(latent)

            fake_video = fake_video.detach().cpu()
            outdata = resizeTensor(fake_video, (args.size_img, args.size_img))

            grid = make_grid(outdata, nrow=args.n_frames, padding=padding)
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

            for j in tqdm(range(args.batch_size), leave=False, desc='Batches'):
                idx1 = (j * args.size_img + j * padding, (j + 1) * args.size_img + j * padding)

                im = Image.fromarray(ndarr[idx1[0]:idx1[1]])
                im.save(f'{save_results}/{i}_{j}.png')

                video = cv2.VideoWriter(f'{save_results}/{i}_{j}.mp4', fourcc, 3, (w_video, h_video))
                for k in tqdm(range(args.n_frames), leave=False, desc='Frames'):
                    idx2 = (k * args.size_img + k * padding, (k + 1) * args.size_img + k * padding)
                    cv_img = cv2.cvtColor(ndarr[idx1[0]:idx1[1], idx2[0]:idx2[1]], cv2.COLOR_RGB2BGR)
                    video.write(cv_img)
                video.release()
            cv2.destroyAllWindows()
