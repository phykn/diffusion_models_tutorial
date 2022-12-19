import os
import torch
from dataclasses import dataclass
from typing import Tuple, Optional
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from src import build_model

@dataclass
class set_args:
    dim: int=32
    init_dim: Optional[int]=None
    out_dim: Optional[int]=None
    dim_mults: Tuple[int]=(1, 2, 4, 8)
    channels: int=1
    self_condition: bool=False
    resnet_block_groups: int=4
    steps: int=1000
    device: str="cuda"

    image_size: int=32
    loss_type: str="huber"

    data_dir: str="data"
    save_dir: str="weight"
    save_model: bool=True
    
    n_epoch: int=20
    batch_size: int=128
    num_workers: int=4
    pin_memory: bool=True
    persistent_workers: bool=True
    lr: float=1e-4
    use_amp: bool=True

def main(args):
    # dataloader
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor()
    ])
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
    ddpm = build_model(args)
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
            x = (x * 2) - 1

            optim.zero_grad()
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    loss = ddpm.loss(x, loss_type=args.loss_type) 
                scaler.scale(loss).backward() 
                scaler.step(optim)
                scaler.update()
            else:
                loss = ddpm.loss(x, loss_type=args.loss_type)
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