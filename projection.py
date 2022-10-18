from datetime import datetime
import sched
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from training_demo import load_progan
from vgg_perceptual_loss import VGGPerceptualLoss
from dataset.real_videos import RealVideos
from visualization.visualizer import saveTensor
from torch.utils.tensorboard import SummaryWriter


torch.manual_seed(0)
import numpy as np
np.random.seed(0)

if __name__ == "__main__":
    now = datetime.now()
    date = now.strftime("%Y.%m.%d_%H.%M.%S")
    name = f'projections_24/baseline_{date}'   # f'{date}_projection'
    import os
    os.mkdir(name)
    log_writer = SummaryWriter(f'runs/{name}')

    progan = load_progan(name='videos_24frames', checkPointDir='output_networks/videos_24frames' , scale=6)

    iters = 800
    bs = 64   # 90
    frames = 24
    w_opt = torch.randn([bs, 512], requires_grad=True, device='cuda')   # .cuda()

    real_images = RealVideos()
    dataloader = DataLoader(real_images, batch_size=bs, shuffle=False)

    mse = nn.MSELoss()
    initial_learning_rate = 0.1   # 0.15
    optimizer = torch.optim.Adam([w_opt], lr=initial_learning_rate)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=300, cycle_momentum=False)

    progan.avgG.eval()
    for real_video in dataloader:   # [bs, 24, 3, 256, 256]
        total_iter = 0
        for frame in range(frames):
            target_image = real_video[:, frame]
            target_image = target_image.cuda()

            for iter in range(iters):
                print(iter)
                optimizer.zero_grad()

                fake_image = progan.avgG(w_opt)
                m_loss = mse(fake_image, target_image)
                log_writer.add_scalar(f'Loss/MSE loss/frame_{frame}', m_loss.item(), iter)
                log_writer.add_scalar(f'Loss/MSE loss', m_loss.item(), total_iter)
                # log_writer.add_scalar(f'LR', scheduler.get_last_lr()[0], total_iter)
                m_loss.backward()
                optimizer.step()
                # scheduler.step()
                total_iter += 1

            iters = 300

            for i in range(10):
                images_comp = torch.cat([target_image[i].unsqueeze(0).cpu(), fake_image[i].unsqueeze(0).cpu()])
                saveTensor(images_comp, (256, 256), f'{name}/{i}_{frame}.jpg')   # {image_name[0][:11]}/{image_name[0]}_{iter}.jpg')
        exit()


        


