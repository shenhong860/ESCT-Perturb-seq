"""DC-W1T with Sun et al. TTT structure: shared cell encoder + main/aux branches."""

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


class CellEnc(nn.Module):
    """Shared feature extractor used by both main and self-supervised branches."""

    def __init__(self, p_dim, dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(p_dim, 256), nn.ReLU(), nn.Linear(256, dim))

    def forward(self, x):
        return self.net(x)


class AuxDec(nn.Module):
    """Self-supervised branch: reconstruct expression from shared cell features."""

    def __init__(self, dim, p_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, p_dim))

    def forward(self, h):
        return self.net(h)


class Velocity(nn.Module):
    def __init__(self, cell_dim, cond_dim, out_dim=1000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cell_dim + 1 + cond_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, out_dim))

    def forward(self, h, t, g):
        return self.net(torch.cat([h, t, g], dim=-1))


class W1Pot(nn.Module):
    def __init__(self, p_dim, emb_dim):
        super().__init__()
        self.f = LBNN(p_dim)
        self.g = nn.Sequential(nn.Linear(emb_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, e):
        return self.f(x) + self.g(e)


class DCW1TTT:
    def __init__(self, p_dim, emb_dim, device, seed=0, w1=True, dir_weight=1.0,
                 aux_weight=1.0, lr=3e-4, ema=0.999):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.p_dim, self.emb_dim, self.device = p_dim, emb_dim, device
        self.dir_weight = dir_weight
        self.aux_weight = aux_weight
        self.ema_decay = ema
        self.cell_enc = CellEnc(p_dim, 128).to(device)
        self.cond_enc = CondEnc(emb_dim).to(device)
        self.vel = Velocity(128, 32, p_dim).to(device)
        self.vel_ema = copy.deepcopy(self.vel)
        self.aux_dec = AuxDec(128, p_dim).to(device)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.w1 = w1
        self.pot = W1Pot(p_dim, emb_dim).to(device) if w1 else None
        self.opt_pot = torch.optim.Adam(self.pot.parameters(), lr=1e-3) if w1 else None

    def parameters(self):
        return (list(self.cell_enc.parameters()) + list(self.cond_enc.parameters()) +
                list(self.vel.parameters()) + list(self.aux_dec.parameters()))

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

    def train_step(self, x0, x1, e, d_target=None):
        g = self.g(e)
        delta = x1 - x0
        t = torch.rand(len(x0), 1, device=self.device)
        s = torch.clamp(t + 0.05, max=1.0)
        xt = x0 + t * delta
        xs = x0 + s * delta
        h0 = self.cell_enc(xt)
        hs = self.cell_enc(xs)
        vt = self.vel(h0, t, g)
        with torch.no_grad():
            vs = self.vel_ema(hs, s, g)
            fs = xs + (1.0 - s) * vs
        ft = xt + (1.0 - t) * vt
        norm = delta.norm(dim=1, keepdim=True) + 1e-6
        loss = ((vt - delta) ** 2).mean() + \
               ((ft - fs) ** 2).mean() + ((vt - vs) ** 2).mean()
        if self.dir_weight > 0:
            if d_target is None:
                dmean = (delta / norm).mean(0)
                dmean = dmean / (dmean.norm() + 1e-6)
            else:
                dmean = d_target
            w = norm.squeeze(1) / norm.sum()
            loss = loss - self.dir_weight * (
                (vt / (vt.norm(dim=1, keepdim=True) + 1e-6) * dmean).sum(1) * w).mean()
        aux = ((self.aux_dec(self.cell_enc(x0)) - x0) ** 2).mean()
        loss = loss + self.aux_weight * aux
        pot = None
        if self.w1:
            dh = self.w1_dir(x0, e)
            dt = delta / norm
            loss = loss + 0.1 * ((dt - dh) ** 2).mean()
            h0l = self.pot(x0.detach(), e.detach()).mean()
            h1l = self.pot(x1.detach(), e.detach()).mean()
            pot = -(h0l - h1l)
            self.opt_pot.zero_grad()
            pot.backward()
            self.opt_pot.step()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.ema_update()
        return loss.item(), (pot.item() if pot is not None else None)

    def freeze_for_ttt(self):
        for p in self.cond_enc.parameters():
            p.requires_grad = False
        for p in self.vel.parameters():
            p.requires_grad = False
        if self.pot is not None:
            for p in self.pot.parameters():
                p.requires_grad = False
            self.opt_pot = None
        self.w1 = False
        self.opt = torch.optim.Adam(
            list(self.cell_enc.parameters()) + list(self.aux_dec.parameters()),
            lr=1e-3, weight_decay=1e-4)

    def ttt_adapt(self, control, iters=200, batch=128, mask_frac=0.3):
        rng = np.random.RandomState(2026)
        for _ in range(iters):
            idx = rng.choice(len(control), batch, replace=True)
            xb = torch.as_tensor(control[idx], device=self.device, dtype=torch.float32)
            masked = xb.clone()
            mask = torch.rand(xb.shape, device=self.device) < mask_frac
            masked[mask] = 0.0
            h = self.cell_enc(masked)
            rec = self.aux_dec(h)
            loss = ((rec - xb) ** 2).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

    def predict(self, x, e):
        with torch.no_grad():
            g = self.cond_enc(torch.as_tensor(e, device=self.device, dtype=torch.float32)
                              .unsqueeze(0).expand(len(x), -1))
            xt = torch.as_tensor(x, device=self.device, dtype=torch.float32)
            h = self.cell_enc(xt)
            t = torch.zeros(len(x), 1, device=self.device)
            return xt + self.vel(h, t, g)
