import os
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.utils.utils import loadmodule, getLastCheckPoint, getNameAndPackage, parse_state_name
from models.frame_seed_generator import FrameSeedGenerator
from models.video_discriminator import VideoDiscriminator
from dataset.real_videos import RealVideos
from visualization.visualizer import saveTensor


def load_progan(name='jelito3d_batchsize8', checkPointDir='output_networks/jelito3d_batchsize8'):
    checkpointData = getLastCheckPoint(checkPointDir, name, scale=None, iter=None)
    modelConfig, pathModel, _ = checkpointData
    _, scale, _ = parse_state_name(pathModel)

    module = 'PGAN'
    packageStr, modelTypeStr = getNameAndPackage(module)
    modelType = loadmodule(packageStr, modelTypeStr)

    with open(modelConfig, 'rb') as file:
        config = json.load(file)

    model = modelType(useGPU=True, storeAVG=True, **config)
    model.load(pathModel)

    # sanity check
    for param in model.avgG.parameters():
        assert not (param.requires_grad)

    # freeze discriminator weights
    for param in model.netD.parameters():
        param.requires_grad = False
        assert not (param.requires_grad)

    return model


def clip_singular_value(A):
    U, s, Vh = torch.linalg.svd(A, full_matrices=False)
    s[s > 1] = 1
    return U @ torch.diag(s) @ Vh


if __name__ == "__main__":
    now = datetime.now()
    date = now.strftime("%Y.%m.%d_%H.%M.%S")
    log_writer = SummaryWriter(f'runs/video-GAN_{date}')
    os.mkdir(f'fakes/{date}')
    os.mkdir(f'checkpoints/{date}')

    fsg = FrameSeedGenerator().cuda()
    n_frames = 8
    time = torch.arange(n_frames).unsqueeze(1).cuda()

    progan = load_progan()

    vdis = VideoDiscriminator().cuda()

    criterion = nn.L1Loss()
    params = list(fsg.parameters()) + list(vdis.parameters())
    optimizer = optim.RMSprop(params, lr=0.00005)

    real_videos = RealVideos()
    dataloader = DataLoader(real_videos, batch_size=None, shuffle=True)

    epochs = 50

    fsg.train()
    vdis.train()
    for epoch in range(epochs):
        print(f'epoch: {epoch}')
        total_loss = 0
        dis_loss = 0
        gen_loss = 0
        for real_video in dataloader:
            seeds = torch.rand([1, 2047]).tile(n_frames, 1).cuda()
            input = fsg(seeds, time)          # (N, 512)
            fake_video = progan.avgG(input)   # (N, CH, H, W)

            optimizer.zero_grad()

            _, fake_latent = progan.netD(fake_video, getFeature=True)   # (N, 512)
            _, real_latent = progan.netD(real_video, getFeature=True)   # (N, 512)
            dis_fake = vdis(fake_latent.permute(1, 0).unsqueeze(0))     # (1, 512, N) -> (1, 1)
            dis_real = vdis(real_latent.permute(1, 0).unsqueeze(0))     # (1, 512, N) -> (1, 1)

            loss = criterion(dis_fake, dis_real)

            total_loss += loss.item()
            dis_loss += loss.item()
            gen_loss += dis_fake.item()

            loss.backward()
            optimizer.step()

            # this can be done less often (once in 5 iterations)
            for param in vdis.parameters():
                # clip weights for convolutions
                if len(param.shape) >= 3:
                    A = param.data.reshape((param.data.shape[0], -1))
                    A = clip_singular_value(A)
                    param.data = A.reshape(param.data.shape)

        
        log_writer.add_scalar('Loss/train', total_loss, epoch)
        log_writer.add_scalar('Discriminator loss/train', dis_loss, epoch)
        log_writer.add_scalar('Generator loss/train', gen_loss, epoch)
        torch.save(fsg.state_dict(), f'checkpoints/{date}/frame_seed_generator.pt')
        torch.save(vdis.state_dict(), f'checkpoints/{date}/video_discriminator.pt')

        fake_video = fake_video.detach().cpu()
        os.mkdir(f'fakes/{date}/epoch_{epoch}')
        for i, frame in enumerate(fake_video):
            saveTensor(frame.unsqueeze(0), (1024, 1024), f'fakes/{date}/epoch_{epoch}/frame_{i}.jpg')

        # save video
        # fake_video = fake_video.detach().cpu()
        # fake_video = fake_video.reshape(n_frames, 1024, 1024, 3)

        # fake_video = np.asarray(fake_video) # * 255
        # print(np.min(fake_video), np.max(fake_video))
        # fake_video.dtype = np.uint8
        # fake_video = list(fake_video)
        # clip = ImageSequenceClip(fake_video, color=[R, G, B], fps=4)
        # clip.write_videofile(f"/home/z1143165/video-GAN/fakes/epoch_{epoch}.mp4", fps=4, audio=False)



