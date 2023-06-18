from datetime import datetime
import os
from math import sqrt
import torch
from torch.utils.data import DataLoader
import torchvision
from PIL import Image

from torch.linalg import vector_norm, matrix_norm
from training_demo import load_progan
from dataset.consec_frames import RealVideos
from visualization.visualizer import saveTensor
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM


torch.manual_seed(0)
import numpy as np
np.random.seed(0)


LATENT_DIM = 16*16*3


if __name__ == "__main__":
    now = datetime.now()
    date = now.strftime("%Y.%m.%d_%H.%M.%S")
    name = f'{date}_16x16'
    log_writer = SummaryWriter(f'logs/{name}')
    os.mkdir(f'checkpoints/{name}')

    device = 'cuda'
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt2 = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id, n_embd=LATENT_DIM, n_head=12, ignore_mismatched_sizes=True)
    gpt2.to(device)

    epochs = 300
    bs = 256
    dataset = RealVideos()
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(gpt2.parameters())

    gpt2.train()
    
    iter = 0
    for epoch in range(epochs):
        for frames in dataloader:
            input = frames[:, :11].to(device)         # (bs, seq_len, ch, h, w) = [bs, 11, 3, 16, 16]
            target_frame = frames[:, -1].to(device)   # (bs, ch, h, w) = [bs, 3, 16, 16]

            optimizer.zero_grad()

            # 1. flatten frames
            input = input.flatten(start_dim=2)        # [bs, 11, 3, 16, 16] -> [bs, 11, 768]

            # 2. predict next frame
            output = gpt2(inputs_embeds=input, output_hidden_states=True)
            output = output.hidden_states[-1]         # hidden state of last layer for all words   [bs, 11, 768]
            noise = output[:, -1]                     # predicted last frame   [bs, 768]

            # 3. unflatten predicted frame
            predicted_frame = noise.reshape([-1, 3, 16, 16])
            assert predicted_frame.shape == target_frame.shape
            loss = mse(predicted_frame, target_frame)
            log_writer.add_scalar(f'Loss per batch', loss.detach().item(), iter)
            iter += 1
            loss.backward()
            optimizer.step()
        
        # visualize
        target_frame = target_frame[0]
        predicted_frame = predicted_frame[0]
        grid = torchvision.utils.make_grid([target_frame.cpu(), predicted_frame.cpu()])
        grid += 1
        images_comp = grid.mul(255).clamp_(0, 255).to("cpu", torch.uint8)
        log_writer.add_image('Target vs. predicted', images_comp, global_step=epoch)

        # save checkpoint
        if (epoch+1) % 50 == 0:
            torch.save(gpt2.state_dict(), f'checkpoints/{name}/gpt2.pt')
            torch.save(optimizer.state_dict(), f'checkpoints/{name}/optim.pt')
