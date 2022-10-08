import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict
from tqdm import tqdm


def ddpm_schedules(
    beta1: float, 
    beta2: float, 
    T: int
) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    # linear function from beta1 to beta2 in timestep T
    beta_t = (beta2 - beta1) * torch.arange(0, T + 1).float() / T + beta1

    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(
        self, 
        model: nn.Module, 
        betas: Tuple[float], 
        n_T: int, 
        drop_prob: float=0.1
    ) -> None:
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device
                
        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v.to(self.device))

        self.n_T = n_T
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(
        self, 
        x: torch.Tensor, 
        c: torch.Tensor
    ) -> torch.Tensor:
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T, (x.size(0),)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = self.sqrtab[_ts, None, None, None]*x + self.sqrtmab[_ts, None, None, None]*noise
        # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device)

        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.model(x_t, c, _ts/self.n_T, context_mask))

    @torch.no_grad()
    def sample(
        self, 
        n_sample: int,
        size: Tuple[int],
        n_classes: int=10, 
        guide_w: float=0.0
    ) -> Tuple[torch.Tensor, np.ndarray]:
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        n_repeat = int(n_sample / n_classes)
        B = n_repeat * n_classes
        x_i = torch.randn(B, *size).to(self.device)  # x_T ~ N(0, 1), sample initial noise
        c_i = torch.arange(0, n_classes).to(self.device) # context for us just cycles throught the mnist labels
        c_i = c_i.repeat(n_repeat)

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(self.device)

        # double the batch
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[B:] = 1. # makes second half of batch context free

        x_i_store = [] # keep track of generated steps in case want to plot something

        # loop
        for curr_t in tqdm(range(self.n_T, 0, -1)):
            t_is = torch.tensor([curr_t / self.n_T]).to(self.device)
            t_is = t_is.repeat(B, 1, 1, 1)

            # double batch
            x_i = x_i.repeat(2, 1, 1, 1)
            t_is = t_is.repeat(2, 1, 1, 1)

            z = torch.randn(B, *size).to(self.device) if curr_t > 1 else 0

            # split predictions and compute weighting
            eps = self.model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:B]
            eps2 = eps[B:]
            eps = (1+guide_w) * eps1 - guide_w * eps2
            x_i = x_i[:B]
            x_i = self.oneover_sqrta[curr_t] * (x_i-eps*self.mab_over_sqrtmab[curr_t]) + self.sqrt_beta_t[curr_t] * z

            x_i_store.append(x_i.detach().cpu().numpy())
        return x_i, np.array(x_i_store)