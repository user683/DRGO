import torch
import torch.nn as nn
from torchsde import sdeint


class VP_SDE(nn.Module):
    def __init__(self, hid_dim, device, beta_min=0.1, beta_max=20 , dt=1e-2):
        super(VP_SDE, self).__init__()
        self.hid_dim = hid_dim
        self.beta_min, self.beta_max = beta_min, beta_max
        self.dt = dt
        self.device = device

        self.score_fn = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, hid_dim)
        )
        for w in self.score_fn:
            if isinstance(w, nn.Linear):
                nn.init.xavier_uniform_(w.weight)

    def calc_score(self, x):
        return self.score_fn(x)

    def forward_sde(self, x, t):
        def f(_t, y):
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            return -0.5 * beta_t * y

        def g(_t, y):
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            bs = y.size(0)
            noise = torch.ones((bs, self.hid_dim, 1), device=self.device)
            return (beta_t ** 0.5) * noise

        ts = torch.Tensor([0, t]).to(self.device)
        output = sdeint(SDEWrapper(f, g), y0=x, ts=ts, dt=self.dt)[-1]
        return output

    def reverse_sde(self, x, t):
        def f(_t, y):
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            score = self.calc_score(y)
            torch.cuda.empty_cache()
            print(beta_t)
            print(y)
            drift = -0.5 * beta_t * y - beta_t * score
            return drift

        def g(_t, y):
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            bs = y.size(0)
            noise = torch.ones((bs, self.hid_dim, 1), device=y.device)
            return (beta_t ** 0.5) * noise

        ts = torch.Tensor([0, t]).to(self.device)
        output = sdeint(SDEWrapper(f, g), y0=x, ts=ts, dt=self.dt)[-1]
        return output

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        log_mean_coeff = torch.Tensor([log_mean_coeff]).to(self.device)
        mean = torch.exp(log_mean_coeff.unsqueeze(-1)) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std


class SDEWrapper(nn.Module):
    sde_type = 'stratonovich'
    noise_type = 'scalar'

    def __init__(self, f, g):
        super(SDEWrapper, self).__init__()
        self.f, self.g = f, g

    def f(self, t, y):
        return self.f(t, y)

    def g(self, t, y):
        return self.g(t, y)
