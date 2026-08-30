"""DC-W1T model: W1/Lipschitz transport + directional coherence loss."""

import copy

import numpy as np
import torch
import torch.nn as nn


def cayley(w):
    if len(w.shape) == 2:
        return cayley(w[None])[0]
    _, cout, cin = w.shape
    if cin > cout:
        return cayley(w.transpose(1, 2)).transpose(1, 2)
    u, v = w[:, :cin], w[:, cin:]
    eye = torch.eye(cin, dtype=w.dtype, device=w.device)[None, :, :]
    a = u - u.transpose(1, 2).conj() + v.transpose(1, 2).conj() @ v
    inv = torch.linalg.solve(eye + a, eye)
    return torch.cat((inv @ (eye - a), -2 * v @ inv), dim=1)


class GroupSort(nn.Module):
    def __init__(self, groups=2):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        b, c = x.shape
        x = x.view(b, c // self.groups, self.groups)
        x, _ = x.sort(dim=-1)
        return x.view(b, c)


class CayleyLayer(nn.Linear):
    def __init__(self, dim_in, dim_out, groups=1):
        super().__init__(dim_in, dim_out)
        self.alpha = nn.Parameter(torch.ones(1))
        self.alpha.data = self.weight.norm()
        self.act = nn.Identity() if groups == dim_out else GroupSort(groups)

    def forward(self, x):
        q = cayley(self.alpha * self.weight / (self.weight.norm() + 1e-8))
        return self.act(nn.functional.linear(x, q, self.bias))


class LBNN(nn.Module):
    def __init__(self, dim_in, hidden=(64, 64), groups=2):
        super().__init__()
        layers = []
        prev = dim_in
        for h in hidden:
            layers.append(CayleyLayer(prev, h, groups))
            prev = h
        layers.append(CayleyLayer(prev, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CondEnc(nn.Module):
    def __init__(self, dim_in, dim_out=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim_in, 32), nn.ReLU(), nn.Linear(32, dim_out))

    def forward(self, x):
        return self.net(x)


class Velocity(nn.Module):
    def __init__(self, p_dim, cond_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(p_dim + 1 + cond_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, p_dim))

    def forward(self, x, t, g):
        return self.net(torch.cat([x, t, g], dim=-1))


class W1Pot(nn.Module):
    def __init__(self, p_dim, emb_dim):
        super().__init__()
        self.f = LBNN(p_dim)
        self.g = nn.Sequential(nn.Linear(emb_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, e):
        return self.f(x) + self.g(e)


class DCW1T:
    def __init__(self, p_dim, emb_dim, device, seed=0, w1=True, dir_weight=1.0,
                 lr=3e-4, ema=0.999):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.p_dim, self.emb_dim, self.device = p_dim, emb_dim, device
        self.dir_weight = dir_weight
        self.ema_decay = ema
        self.cond_enc = CondEnc(emb_dim).to(device)
        self.vel = Velocity(p_dim, 32).to(device)
        self.vel_ema = copy.deepcopy(self.vel)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.w1 = w1
        self.pot = W1Pot(p_dim, emb_dim).to(device) if w1 else None
        self.opt_pot = torch.optim.Adam(self.pot.parameters(), lr=1e-3) if w1 else None

    def parameters(self):
        return list(self.cond_enc.parameters()) + list(self.vel.parameters())

    def g(self, e):
        return self.cond_enc(e)

    def ema_update(self):
        for p, pe in zip(self.vel.parameters(), self.vel_ema.parameters()):
            pe.data.mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay)

    def w1_dir(self, x0, e):
        xg = x0.detach().clone().requires_grad_(True)
        h = self.pot(xg, e.detach())
        grad = torch.autograd.grad(h.sum(), xg, retain_graph=False)[0].detach()
        return grad / (grad.norm(dim=1, keepdim=True) + 1e-6)

    def train_step(self, x0, x1, e):
        g = self.g(e)
        delta = x1 - x0
        t = torch.rand(len(x0), 1, device=self.device)
        s = torch.clamp(t + 0.05, max=1.0)
        xt = x0 + t * delta
        xs = x0 + s * delta
        vt = self.vel(xt, t, g)
        with torch.no_grad():
            vs = self.vel_ema(xs, s, g)
            fs = xs + (1.0 - s) * vs
        ft = xt + (1.0 - t) * vt
        loss = ((vt - delta) ** 2).mean() + \
               ((ft - fs) ** 2).mean() + ((vt - vs) ** 2).mean()
        norm = delta.norm(dim=1, keepdim=True) + 1e-6
        if self.dir_weight > 0:
            dmean = (delta / norm).mean(0)
            dmean = dmean / (dmean.norm() + 1e-6)
            loss = loss - self.dir_weight * (
                (vt / (vt.norm(dim=1, keepdim=True) + 1e-6) * dmean).sum(1) *
                (norm.squeeze(1) / norm.sum())).mean()
        pot = None
        if self.w1:
            dh = self.w1_dir(x0, e)
            dt = delta / norm
            loss = loss + 0.1 * ((dt - dh) ** 2).mean()
            h0 = self.pot(x0.detach(), e.detach()).mean()
            h1 = self.pot(x1.detach(), e.detach()).mean()
            pot = -(h0 - h1)
            self.opt_pot.zero_grad()
            pot.backward()
            self.opt_pot.step()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.ema_update()
        return loss.item(), (pot.item() if pot is not None else None)

    def freeze(self):
        for p in self.vel.parameters():
            p.requires_grad = False
        for p in self.vel.net[-1].parameters():
            p.requires_grad = True
        for p in self.cond_enc.parameters():
            p.requires_grad = True
        self.opt = torch.optim.Adam([p for p in self.parameters() if p.requires_grad],
                                    lr=3e-4)
        self.dir_weight = 0.0
        self.w1 = False
        if self.pot is not None:
            for p in self.pot.parameters():
                p.requires_grad = False
            self.opt_pot = None

    def adapt_identity(self, x0, e, iters=200, batch=128):
        for _ in range(iters):
            idx = np.random.choice(len(x0), batch)
            xb = torch.as_tensor(x0[idx], device=self.device, dtype=torch.float32)
            eb = torch.as_tensor(np.tile(e, (batch, 1)), device=self.device,
                                 dtype=torch.float32)
            g = self.g(eb)
            t = torch.rand(batch, 1, device=self.device)
            vt = self.vel(xb, t, g)
            loss = (vt ** 2).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.ema_update()

    def predict(self, x, e):
        with torch.no_grad():
            g = self.g(torch.as_tensor(e, device=self.device, dtype=torch.float32)
                       .unsqueeze(0).expand(len(x), -1))
            t = torch.zeros(len(x), 1, device=self.device)
            xt = torch.as_tensor(x, device=self.device, dtype=torch.float32)
            return xt + self.vel(xt, t, g)
