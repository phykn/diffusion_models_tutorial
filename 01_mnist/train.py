''' 
This script does conditional image generation on MNIST, using a diffusion model

This code is modified from,
https://github.com/cloneofsimo/minDiffusion
https://github.com/TeaPearce/Conditional_Diffusion_MNIST

Diffusion model is based on DDPM,
https://arxiv.org/abs/2006.11239

The conditioning idea is taken from 'Classifier-Free Diffusion Guidance',
https://arxiv.org/abs/2207.12598

This technique also features in ImageGen 'Photorealistic Text-to-Image Diffusion Modelswith Deep Language Understanding',
https://arxiv.org/abs/2205.11487
'''

import os
import torch
from typing import Tuple
from dataclasses import dataclass
from tqdm import tqdm
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from src.unet import ContextUnet
from src.ddpm import DDPM

@dataclass
class set_args:
    data_dir: str="data"
    save_dir: str="weight"
    save_model: bool=True
    
    n_classes: int=10
    n_feat: int=128
    n_T: int=400
    betas: Tuple[float]=(1e-4, 0.02)
    drop_prob: float=0.1

    n_epoch: int=20
    batch_size: int=128
    num_workers: int=4
    pin_memory: bool=True
    persistent_workers: bool=True
    lr: float=1e-4
    device: str="cuda"
    use_amp: bool=True

    ws_test: Tuple[float]=(0.0, 0.5, 2.0)


def main(args):
    # dataloader
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST(args.data_dir, train=True, download=True, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers
    )

    # model
    model = ContextUnet(
        in_channels=1, 
        n_feat=args.n_feat, 
        n_classes=args.n_classes
    ).to(args.device)

    ddpm = DDPM(
        model=model, 
        betas=args.betas, 
        n_T=args.n_T, 
        drop_prob=args.drop_prob
    )

    optim = torch.optim.Adam(ddpm.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    # loop
    for ep in range(args.n_epoch):
        ddpm.train()

        # linear lr decay
        optim.param_groups[0]["lr"] = args.lr * (1-ep/args.n_epoch)

        # train
        loss_ema = None
        pbar = tqdm(dataloader, desc=f"EPOCH {ep+1:03d}")

        for x, c in pbar:
            x = x.to(args.device)
            c = c.to(args.device)

            optim.zero_grad()
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    loss = ddpm(x, c) 
                scaler.scale(loss).backward() 
                scaler.step(optim)
                scaler.update()
            else:
                loss = ddpm(x, c)
                loss.backward()
                optim.step()

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_postfix(dict(loss=loss_ema))
        
        # save model
        if args.save_model:
            os.makedirs(args.save_dir, exist_ok=True)
            path = os.path.join(args.save_dir, f"model_{ep:03d}.pth")
            torch.save(ddpm.state_dict(), path)
            print(f"saved model at {path}")

if __name__ == "__main__":
    args = set_args()
    main(args)