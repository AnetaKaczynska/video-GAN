import os
import json
from datetime import datetime

from tqdm import tqdm
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

torch.autograd.set_detect_anomaly(True)


def load_progan(name='jelito3d_batchsize8', checkPointDir='/shared/results/z1143165/jelito3d_batchsize8'):
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
    os.mkdir(f'reals/{date}')
    os.mkdir(f'checkpoints/{date}')

    fsg = FrameSeedGenerator().cuda()
    n_frames = 24
    time = torch.arange(n_frames).unsqueeze(1).cuda()

    progan = load_progan()

    vdis = VideoDiscriminator(True).cuda()

    criterion = nn.BCELoss()
    lr = 0.0002
    beta1 = 0.5
    optimizer_G = optim.Adam(fsg.parameters(), lr=lr, betas=(beta1, 0.999))   # RMSprop(fsg.parameters(), lr=0.00005)
    optimizer_D = optim.Adam(vdis.parameters(), lr=lr, betas=(beta1, 0.999))   # RMSprop(vdis.parameters(), lr=0.00005)
    # optimizer_D = optim.Adam([vdis.parameters() + progan.parameters()], lr=lr, betas=(beta1, 0.999))

    real_videos = RealVideos()
    dataloader = DataLoader(real_videos, batch_size=None, shuffle=True)

    epochs = 50
    batch_size = 1

    fsg.train()
    vdis.train()
    progan.netD.eval()
    progan.avgG.eval()
    for epoch in range(epochs):
        print(f'epoch: {epoch}')
        dis_loss = 0
        gen_loss = 0
        total = 0
        dis_out_fakes = 0
        dis_out_reals = 0
        acc_fakes = 0
        acc_reals = 0
        #for iter, real_video in tqdm(enumerate(dataloader)):
        for iter, real_video in enumerate(dataloader):
            real_video = real_video.cuda()

            # update discriminator
            optimizer_D.zero_grad()
            # - real input
            _, real_latent = progan.netD(real_video, getFeature=True)   # (N, 512)
            real_latent = real_latent.unsqueeze(0)                      # (1, N, 512)
            dis_real = vdis(real_latent)                                # (1, 1)
            label = torch.full([batch_size], 1., dtype=torch.float).cuda()
            errD_real = criterion(dis_real.squeeze(0), label)
            errD_real.backward()
            D_x = dis_real.mean().item()

            # - fake input
            noise = torch.rand([1, 2047]).tile(n_frames, 1).cuda()
            pro_noise = torch.rand([1, 512]).cuda()
            input = fsg(noise, pro_noise, time)
            fake_video = progan.avgG(input)
            _, fake_latent = progan.netD(fake_video.detach(), getFeature=True)   # (N, 512)
            fake_latent = fake_latent.unsqueeze(0)                               # (1, N, 512)
            dis_fake = vdis(fake_latent)                                         # (1, 1)
            label.fill_(0.)
            errD_fake = criterion(dis_fake.squeeze(0), label)
            errD_fake.backward()
            # for param in vdis.parameters():
            #     print(param.grad)
            # exit()
            D_G_z1 = dis_fake.mean().item()
            errD = errD_real + errD_fake
            optimizer_D.step()

            dis_out_fakes += D_G_z1
            dis_out_reals += D_x
            acc_fakes += 1 if round(D_G_z1) == 0 else 0
            acc_reals += 1 if round(D_x) == 1 else 0

            if epoch > 10:
                # update generator
                optimizer_G.zero_grad()
                # do we have to do this 2 lines again?
                _, fake_latent = progan.netD(fake_video, getFeature=True)   # (N, 512)
                dis_fake = vdis(fake_latent.unsqueeze(0))                   # (1, 512, N) -> (1, 1)
                label.fill_(1.)
                errG = criterion(dis_fake.squeeze(0), label)
                errG.backward()
                D_G_z2 = dis_fake.mean().item()
                optimizer_G.step()
            else:
                errG = torch.Tensor([0])

            dis_loss += errD.item()
            gen_loss += errG.item()
            total += 1

            if (iter+1) % 1000 == 0:
                step = len(dataloader)*epoch+iter
                log_writer.add_scalar('Discriminator loss', dis_loss/total, step)
                log_writer.add_scalar('Generator loss', gen_loss/total, step)
                log_writer.add_scalar('Discriminator output/fakes', dis_out_fakes/total, step)
                log_writer.add_scalar('Discriminator output/reals', dis_out_reals/total, step)
                log_writer.add_scalar('Accuracy/fakes', acc_fakes/total, step)
                log_writer.add_scalar('Accuracy/reals', acc_reals/total, step)
                dis_loss = 0
                gen_loss = 0
                total = 0
                dis_out_fakes = 0
                dis_out_reals = 0
                acc_fakes = 0
                acc_reals = 0

            if iter % 1000 == 0:
                fake_video = fake_video.detach().cpu()
                #real_video = real_video.detach().cpu()
                saveTensor(fake_video, (1024, 1024), f'fakes/{date}/video_{epoch}_{iter}.jpg')
                #saveTensor(real_video, (1024, 1024), f'reals/{date}/video_{epoch}_{iter}.jpg')
        
        torch.save(fsg.state_dict(), f'checkpoints/{date}/frame_seed_generator.pt')
        torch.save(vdis.state_dict(), f'checkpoints/{date}/video_discriminator.pt')





