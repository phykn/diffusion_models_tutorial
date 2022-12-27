import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import Unet
from .schedule import cosine_beta_schedule as noise_schedule

class DDIM(nn.Module):
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

    def taus(self, step):
        self.schedules(step)
        ts = list(range(0, step))
        return ts[:-1], ts[1:]

    @torch.no_grad()
    def p_epsilon(self, x, t):
        return self.unet(x, t)

    def forward_xt(self, x, t, noise):
        sqrt_alphas_cumprod_t = self.extract(
            self.sqrt_alphas_cumprod, t, x.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_x0(self, x, t, noise):
        sqrt_alphas_cumprod_t = self.extract(
            self.sqrt_alphas_cumprod, t, x.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        return (x - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t

    def p_sample_loop(self, shape, step=1):
        b = shape[0]
        x = torch.randn(shape, device=self.device)

        imgs = []
        ts_prev, ts_curr = self.taus(step)
        for t_prev, t_curr in zip(reversed(ts_prev), reversed(ts_curr)):
            t = torch.full((b,), t_curr, device=self.device, dtype=torch.long)
            epsilon = self.p_epsilon(x, t)
            x = self.predict_x0(x, t, epsilon)
            x = torch.clip(x, -1, 1)
            t = torch.full((b,), t_prev, device=self.device, dtype=torch.long)
            x = self.forward_xt(x, t, epsilon)
            imgs.append(x.clone().cpu().numpy())
        return imgs

    def sample(self, image_size, batch_size=16, step=1):
        shape = (batch_size, self.channels, image_size, image_size)
        return self.p_sample_loop(shape, step)