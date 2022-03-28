from os import O_SYNC


import os
import sys
import json

from moviepy.editor import ImageSequenceClip
import numpy as np
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


def load_progan():
    name = 'jelito3d_batchsize8'
    checkPointDir = '/home/z1143165/video-GAN/output_networks/jelito3d_batchsize8'
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

    return model


if __name__ == "__main__":
    log_writer = SummaryWriter('/home/z1143165/video-GAN/runs')

    fsg = FrameSeedGenerator().cuda()
    n_frames = 8
    time = torch.arange(n_frames).unsqueeze(1)

    progan = load_progan()

    vdis = VideoDiscriminator(3, 4, 64).cuda()

    criterion = nn.L1Loss()
    params = list(fsg.parameters()) + list(vdis.parameters())
    optimizer = optim.RMSprop(params, lr=0.00005)

    real_videos = RealVideos()
    dataloader = DataLoader(real_videos, batch_size=1, shuffle=True)

    epochs = 10

    fsg.train()
    vdis.train()
    for epoch in range(epochs):
        total_loss = 0
        for real_video in dataloader:
            seeds = torch.rand([n_frames, 2047])
            input = torch.hstack([time, seeds]).cuda()

            input = fsg(input)

            output = progan.avgG(input)

            fake_video = output.reshape([1, 3, n_frames, 1024, 1024])   # (N, CH, T, H, W)
            real_video = real_video.cuda()

            optimizer.zero_grad()

            dis_fake = vdis(fake_video)
            dis_real = vdis(real_video)
            loss = criterion(dis_fake, dis_real)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            break
        
        log_writer.add_scalar('Loss/train', total_loss, epoch)
        torch.save(fsg.state_dict(), '/home/z1143165/video-GAN/checkpoints/frame_seed_generator.pt')
        torch.save(vdis.state_dict(), '/home/z1143165/video-GAN/checkpoints/video_discriminator.pt')

        fake_video = fake_video.detach().cpu().reshape(n_frames, 3, 1024, 1024)
        os.mkdir(f'fakes/epoch_{epoch}')
        for i, frame in enumerate(fake_video):
            saveTensor(frame.unsqueeze(0), (1024, 1024), f'fakes/epoch_{epoch}/frame_{i}.jpg')

        # save video
        # fake_video = fake_video.detach().cpu()
        # fake_video = fake_video.reshape(n_frames, 1024, 1024, 3)

        # fake_video = np.asarray(fake_video) # * 255
        # print(np.min(fake_video), np.max(fake_video))
        # fake_video.dtype = np.uint8
        # fake_video = list(fake_video)
        # clip = ImageSequenceClip(fake_video, color=[R, G, B], fps=4)
        # clip.write_videofile(f"/home/z1143165/video-GAN/fakes/epoch_{epoch}.mp4", fps=4, audio=False)



