"""Train a conditional DDPM on MNIST and save the checkpoint.

Adapted from TeaPearce/Conditional_Diffusion_MNIST (MIT License)
https://github.com/TeaPearce/Conditional_Diffusion_MNIST

Usage:
    python scripts/train_mnist_ddpm.py --n_epoch 20 --output models/mnist_ddpm/ddpm_mnist_20ep.pth
"""

import argparse
import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import numpy as np


# === Model Definition ===

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_res=False):
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels), nn.GELU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels), nn.GELU())

    def forward(self, x):
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            out = (x + x2) if self.same_channels else (x1 + x2)
            return out / 1.414
        return self.conv2(self.conv1(x))


class UnetDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.model = nn.Sequential(ResidualConvBlock(in_ch, out_ch), nn.MaxPool2d(2))
    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, 2),
            ResidualConvBlock(out_ch, out_ch),
            ResidualConvBlock(out_ch, out_ch))
    def forward(self, x, skip):
        return self.model(torch.cat((x, skip), 1))


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim))
    def forward(self, x):
        return self.model(x.view(-1, self.input_dim))


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=128, n_classes=10):
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_classes, n_feat)
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7),
            nn.GroupNorm(8, 2 * n_feat), nn.ReLU())
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat), nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1))

    def forward(self, x, c, t, context_mask):
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)
        c = F.one_hot(c, num_classes=self.n_classes).float()
        context_mask = (-1 * (1 - context_mask[:, None].repeat(1, self.n_classes)))
        c = c * context_mask
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 * up1 + temb1, down2)
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        return self.out(torch.cat((up3, x), 1))


def ddpm_schedules(beta1, beta2, T):
    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    alpha_t = 1 - beta_t
    alphabar_t = torch.cumsum(torch.log(alpha_t), dim=0).exp()
    return {
        'alpha_t': alpha_t,
        'oneover_sqrta': 1 / torch.sqrt(alpha_t),
        'sqrt_beta_t': torch.sqrt(beta_t),
        'alphabar_t': alphabar_t,
        'sqrtab': torch.sqrt(alphabar_t),
        'sqrtmab': torch.sqrt(1 - alphabar_t),
        'mab_over_sqrtmab': (1 - alpha_t) / torch.sqrt(1 - alphabar_t),
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super().__init__()
        self.nn_model = nn_model.to(device)
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)
        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)
        x_t = self.sqrtab[_ts, None, None, None] * x + self.sqrtmab[_ts, None, None, None] * noise
        context_mask = torch.bernoulli(torch.zeros_like(c, dtype=torch.float32) + self.drop_prob).to(self.device)
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))


# === Training ===

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_T', type=int, default=400)
    parser.add_argument('--n_feat', type=int, default=128)
    parser.add_argument('--lrate', type=float, default=1e-4)
    parser.add_argument('--output', type=str, default='models/mnist_ddpm/ddpm_mnist_20ep.pth')
    parser.add_argument('--data_dir', type=str, default='./data')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    ddpm = DDPM(
        nn_model=ContextUnet(in_channels=1, n_feat=args.n_feat, n_classes=10),
        betas=(1e-4, 0.02), n_T=args.n_T, device=device, drop_prob=0.1
    ).to(device)

    n_params = sum(p.numel() for p in ddpm.parameters())
    print(f'Model parameters: {n_params:,}')

    tf = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST(args.data_dir, train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    optim = torch.optim.Adam(ddpm.parameters(), lr=args.lrate)

    loss_history = []
    for ep in range(args.n_epoch):
        ddpm.train()
        optim.param_groups[0]['lr'] = args.lrate * (1 - ep / args.n_epoch)

        pbar = tqdm(dataloader, desc=f'Epoch {ep+1}/{args.n_epoch}')
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            loss = ddpm(x.to(device), c.to(device))
            loss.backward()
            optim.step()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f'Epoch {ep+1}/{args.n_epoch} | loss: {loss_ema:.4f}')

        loss_history.append(loss_ema)
        print(f'  Epoch {ep+1} done, loss_ema = {loss_ema:.4f}')

    # Save checkpoint with loss history
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        'model_state_dict': ddpm.state_dict(),
        'loss_history': loss_history,
        'n_epoch': args.n_epoch,
        'n_T': args.n_T,
        'n_feat': args.n_feat,
    }, args.output)
    print(f'\nModel saved to {args.output}')
    print(f'File size: {os.path.getsize(args.output) / 1e6:.1f} MB')
    print(f'Loss history: {loss_history}')


if __name__ == '__main__':
    main()
