from datetime import datetime
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from training_demo import load_progan
from vgg_perceptual_loss import VGGPerceptualLoss
from dataset.real_images_tmp import RealImages
from visualization.visualizer import saveTensor
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    now = datetime.now()
    date = now.strftime("%Y.%m.%d_%H.%M.%S")
    name = f'{date}_projection'
    log_writer = SummaryWriter(f'runs/{name}')

    progan = load_progan(scale=6)

    iters = 800
    w_opt = torch.randn([1, 512], requires_grad=True, device='cuda')   # .cuda()

    real_images = RealImages()
    dataloader = DataLoader(real_images, batch_size=1, shuffle=False)

    perceptual_loss = VGGPerceptualLoss()
    mse = nn.MSELoss()
    initial_learning_rate = 0.1
    optimizer = torch.optim.Adam([w_opt], lr=initial_learning_rate)

    progan.avgG.eval()
    for image_name, target_image in dataloader:
        target_image = target_image.cuda()
        for iter in range(iters):
            print(image_name, iter)
            # print('w_opt: ', w_opt[0][:5])
            optimizer.zero_grad()

            fake_image = progan.avgG(w_opt)
            p_loss = torch.Tensor([0]) # perceptual_loss(fake_image.cpu(), target_image.cpu())
            m_loss = mse(fake_image, target_image)
            loss = m_loss # + p_loss
            log_writer.add_scalar('Loss/Perceptual loss', p_loss.item(), iter)
            log_writer.add_scalar('Loss/MSE loss', m_loss.item(), iter)
            log_writer.add_scalar('Loss/Total loss', loss.item(), iter)
            loss.backward()
            optimizer.step()

            if iter%50 == 0:
                images_comp = torch.cat([target_image.cpu(), fake_image.cpu()])
                saveTensor(images_comp, (256, 256), f'projections/aaa/{image_name[0]}_{iter}.jpg')



# odlaczyc perceptual loss
# rysowac loss
# zaczac od fakeowego obrazka a nie prawdziego i od tego ustalonego seeda i zobaczyc czy seed sie cos zmieni
# l1 loss zamiast l2 loss
# poglebic siec

        


