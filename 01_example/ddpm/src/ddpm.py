import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import Unet
from .schedule import cosine_beta_schedule as noise_schedule

class DDPM(nn.Module):
    def __init__(self,
        dim: int,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=4,
        steps=4000,
        device="cpu"
    ):
        super().__init__()
        self.unet = Unet(
            dim, init_dim, out_dim, dim_mults, channels, self_condition, resnet_block_groups
        )
        self.unet.to(device)

        for k, v in self.schedules(steps).items():
            self.register_buffer(k, v.to(device))

        self.channels = channels
        self.steps = steps
        self.device = device

    @staticmethod
    def schedules(steps):
        betas = noise_schedule(steps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        return dict(
            betas=betas,
            alphas=alphas,
            alphas_cumprod=alphas_cumprod,
            alphas_cumprod_prev=alphas_cumprod_prev,
            sqrt_recip_alphas=sqrt_recip_alphas,
            sqrt_alphas_cumprod=sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
            posterior_variance=posterior_variance
        )

    @staticmethod
    def extract(a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward(self, x, t, noise):
        sqrt_alphas_cumprod_t = self.extract(
            self.sqrt_alphas_cumprod, t, x.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        x = sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
        return self.unet(x, t)

    def loss(self, x, loss_type="l1"):
        b, c, h, w = x.shape
        t = torch.randint(0, self.steps, (b, ), device=self.device)
        noise = torch.randn_like(x)
        predicted_noise = self(x, t, noise)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
        
        return loss

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = self.extract(
            self.betas, t, x.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.extract(
            self.sqrt_recip_alphas, t, x.shape
        )
        
        mean = sqrt_recip_alphas_t * (x - betas_t * self.unet(x, t) / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return mean
        else:
            posterior_variance_t = self.extract(
                self.posterior_variance, t, x.shape
            )
            noise = torch.randn_like(x)
            return mean + torch.sqrt(posterior_variance_t) * noise

    def p_sample_loop(self, shape):
        b = shape[0]
        img = torch.randn(shape, device=self.device)

        imgs = []
        for t_index in reversed(range(0, self.steps)):
            t = torch.full((b,), t_index, device=self.device, dtype=torch.long)
            img = self.p_sample(img, t, t_index)
            imgs.append(img.cpu().numpy())
        return imgs

    def sample(self, image_size, batch_size=16):
        shape = (batch_size, self.channels, image_size, image_size)
        return self.p_sample_loop(shape)