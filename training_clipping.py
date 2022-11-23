import json
from datetime import datetime
from pathlib import Path
from random import randint

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.real_videos import RealVideos
from models.frame_seed_generator import FrameSeedGenerator
from models.utils.utils import loadmodule, getLastCheckPoint, getNameAndPackage, parse_state_name
from models.video_discriminator import VideoDiscriminator
from visualization.visualizer import saveTensor

torch.autograd.set_detect_anomaly(True)


def load_progan(name='jelita', checkPointDir='/checkpoints/jelita'):
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


if __name__ == "__main__":

    epochs = 50
    batch_size = 2  # 6, 3
    learning_rate = 5e-5
    lr_proD = 5e-5  # 1e-5
    clamp_num = 0.01  # WGAN clip gradient
    n_frames, duration = 24, 24  # (8, 24, 60) x (8, 24, 24)
    size_img = 256  # 1024
    iter_logger = 250
    iterD = 2
    alpha_similarity = 0  # 0, 0.2
    frozen_proD = False  # True

    seed_ = randint(10, 10000)
    torch.manual_seed(seed_)
    torch.cuda.manual_seed(seed_)

    date = datetime.now().strftime("%y%m%d-%H%M%S")
    root_results = f'/results/{Path(__file__).resolve().stem.removeprefix("training_")}' \
                   f'_frame{n_frames}_duration{duration}_iterD{iterD}_alphaSimil{alpha_similarity}' \
                   f'{"" if frozen_proD else "_proDISC"}_seed{seed_}_{date}'
    print(f'\033[0;33m{root_results}\033[0m')

    log_writer = SummaryWriter(f'{root_results}/tensorboard')
    Path(f'{root_results}/fakes').mkdir(parents=True, exist_ok=True)
    # Path(f'{root_results}/reals').mkdir(parents=True, exist_ok=True)
    Path(f'{root_results}/checkpoints').mkdir(parents=True, exist_ok=True)

    fsg = FrameSeedGenerator().cuda()
    progan = load_progan()
    vdis = VideoDiscriminator(active=False).cuda()

    optimizer_G = optim.RMSprop(fsg.parameters(), lr=learning_rate)
    optimizer_D = optim.RMSprop(vdis.parameters(), lr=learning_rate)

    data_path = Path('/datasets')
    img_dir = data_path / "film24images"
    noise_dir = data_path / "noise"
    dataset = RealVideos(img_dir=str(img_dir), noise_dir=str(noise_dir), num_frame=n_frames, duration=duration)
    dataloader = DataLoader(dataset, num_workers=8, pin_memory=True, batch_size=batch_size, shuffle=True, drop_last=True)

    progan.avgG.eval()
    if frozen_proD:
        progan.netD.eval()
        print('\033[0;32mFrozen proGAN Discriminator\033[0m')
    else:
        progan.netD.train()
        optimizer_proD = optim.RMSprop(progan.netD.parameters(), lr=lr_proD)
        print('\033[0;32mTrain proGAN Discriminator\033[0m')

    fsg.train()
    vdis.train()

    torch.cuda.empty_cache()

    min_gen_similarity = 1.0
    epoch_bar = tqdm(range(epochs), desc='Training')
    for epoch in epoch_bar:
        dis_loss = gen_loss = gen_similarity = total = 0
        dis_out_fakes = dis_out_reals = 0
        acc_fakes = acc_reals = 0

        for iter, (real_video, real_noise, time) in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
            real_video = real_video.cuda()
            real_noise = real_noise.cuda()
            time = time.float().cuda()

            #######################
            # Train discriminator #
            #######################

            for _ in range(iterD):
                # modification: clip param for discriminator
                for parm in vdis.parameters():
                    parm.data.clamp_(-clamp_num, clamp_num)

                optimizer_D.zero_grad()
                if not frozen_proD:
                    optimizer_proD.zero_grad()

                # Pass real images through discriminator
                _, real_latent = progan.netD(real_video.view(-1, *real_video.shape[-3:]), getFeature=True)
                real_latent = real_latent.view(batch_size, duration, -1)  # (bs, N, 512)
                real_preds = vdis(real_latent, time)  # (bs, 1)

                # modification: remove binary cross entropy
                # real_targets = torch.ones(real_images.size(0), 1, device='cuda')
                # real_loss = F.binary_cross_entropy(real_preds, real_targets)
                real_loss = -torch.mean(real_preds)

                # Generate fake images
                noise = torch.randn([batch_size, 2047]).unsqueeze(1).tile(1, duration, 1).view(-1, 2047).cuda()
                latent = fsg(noise, real_noise, time.view(-1, 1))
                fake_video = progan.avgG(latent).detach()

                # Pass fake images through discriminator
                _, fake_latent = progan.netD(fake_video, getFeature=True)  # (bs * N, 512)
                fake_latent = fake_latent.view(batch_size, duration, -1)  # (bs, N, 512)
                fake_preds = vdis(fake_latent, time)  # (bs, 1)

                # modification: remove binary cross entropy
                # fake_targets = torch.zeros(fake_images.size(0), 1, device='cuda')
                # fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
                fake_loss = torch.mean(fake_preds)

                # Update discriminator weights
                loss_D = real_loss + fake_loss
                loss_D.backward()
                optimizer_D.step()

                if not frozen_proD:
                    optimizer_proD.step()

            dis_out_fakes += torch.mean(fake_preds).item()
            dis_out_reals += torch.mean(real_preds).item()
            acc_fakes += torch.ge(fake_preds, 0.).sum().item()
            acc_reals += torch.ge(real_preds, 0.).sum().item()

            ###################
            # Train generator #
            ###################

            optimizer_G.zero_grad()

            # Generate fake images
            noise = torch.randn([batch_size, 2047]).unsqueeze(1).tile(1, duration, 1).view(-1, 2047).cuda()
            latent = fsg(noise, real_noise, time.view(-1, 1))
            fake_video = progan.avgG(latent)

            # Try to fool the discriminator
            _, fake_latent = progan.netD(fake_video, getFeature=True)  # (bs * N, 512)
            fake_latent = fake_latent.view(batch_size, duration, -1)  # (bs, N, 512)
            preds = vdis(fake_latent, time)  # (bs, 1)

            # modificationL remove binary cross entropy
            # targets = torch.ones(batch_size, 1, device='cuda')
            # real_loss = F.binary_cross_entropy(preds, targets)
            loss_G = -torch.mean(preds)

            latent = latent.view(batch_size, duration, -1)
            loss_similarity = torch.norm(torch.roll(latent, shifts=-1, dims=1)[:, :-1, :] - latent[:, :-1, :],
                                         p=2, dim=-1).neg().exp().mean()

            # Update generator weights
            if alpha_similarity > 0:
                (loss_G + alpha_similarity * loss_similarity).backward()
            else:
                loss_G.backward()
            optimizer_G.step()

            epoch_bar.set_description(f'loss_g: {loss_G.item():.4f}, '
                                      f'loss_d: {loss_D.item():.4f}, '
                                      f'real_score: {real_loss.item():.4f}, '
                                      f'fake_score: {fake_loss.item():.4f}')

            dis_loss += loss_D.item()
            gen_loss += loss_G.item()
            gen_similarity += loss_similarity.item()
            total += batch_size

            step = len(dataloader) * epoch + iter
            if step % iter_logger == 0:
                gen_similarity /= iter_logger

                log_writer.add_scalar('Discriminator loss', dis_loss / iter_logger, step)
                log_writer.add_scalar('Generator loss', gen_loss / iter_logger, step)
                log_writer.add_scalar('Generator loss/similarity', gen_similarity, step)
                log_writer.add_scalar('Discriminator output/fakes', dis_out_fakes / iter_logger, step)
                log_writer.add_scalar('Discriminator output/reals', dis_out_reals / iter_logger, step)
                log_writer.add_scalar('Accuracy/fakes', acc_fakes / total, step)
                log_writer.add_scalar('Accuracy/reals', acc_reals / total, step)

                fake_video = fake_video.detach().cpu()
                saveTensor(fake_video, (size_img, size_img), f'{root_results}/fakes/{epoch}_{iter}.jpg', duration)

                # real_video = real_video.reshape(-1, *real_video.shape[-3:])
                # real_video = real_video.detach().cpu()
                # saveTensor(real_video, (size_img, size_img), f'{root_results}/reals/{epoch}_{iter}.jpg', duration)

                if min_gen_similarity > gen_similarity:
                    min_gen_similarity = gen_similarity
                    torch.save({'model_state_dict': fsg.state_dict(), 'epoch': epoch, 'iter': iter},
                               f'{root_results}/checkpoints/frame_seed_generator_maxSimilarity.pt')

                dis_loss = gen_loss = gen_similarity = total = 0
                dis_out_fakes = dis_out_reals = 0
                acc_fakes = acc_reals = 0

        torch.save({'model_state_dict': fsg.state_dict(), 'epoch': epoch},
                   f'{root_results}/checkpoints/frame_seed_generator.pt')
        # torch.save(vdis.state_dict(), f'{root_results}/checkpoints/video_discriminator.pt')