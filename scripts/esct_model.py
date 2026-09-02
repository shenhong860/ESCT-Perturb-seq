"""ESCT model: additive direction prior + residual velocity."""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_blocks import AuxDec, CellEnc, CondEnc, GroupSort, LBNN, W1Pot


def rbf_mmd(x, y, sigma=2.0):
    xx = torch.exp(-torch.cdist(x, x) / (2 * sigma * sigma))
    yy = torch.exp(-torch.cdist(y, y) / (2 * sigma * sigma))
    xy = torch.exp(-torch.cdist(x, y) / (2 * sigma * sigma))
    return (xx.mean() + yy.mean() - 2 * xy.mean())


def sinkhorn_loss(x, y, eps=0.1, iters=20):
    c = torch.cdist(x, y) ** 2
    n, m = len(x), len(y)
    a = torch.ones(n, device=x.device) / n
    b = torch.ones(m, device=y.device) / m
    k = torch.exp(-c / eps)
    u = torch.ones_like(a)
    for _ in range(iters):
        v = b / (k.T @ u + 1e-12)
        u = a / (k @ v + 1e-12)
    p = u[:, None] * k * v[None, :]
    return (p * c).sum()


class ResidualVelocity(nn.Module):
    def __init__(self, cell_dim, cond_dim, out_dim=1000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cell_dim + 1 + cond_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, out_dim))

    def forward(self, h, t, g):
        return self.net(torch.cat([h, t, g], dim=-1))


class GammaGate(nn.Module):
    def __init__(self, cond_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cond_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, g):
        return F.softplus(self.net(g))


class ResGate(nn.Module):
    def __init__(self, cond_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cond_dim, 16), nn.ReLU(), nn.Linear(16, 1))
        self.net[-1].bias.data.fill_(-3.0)

    def forward(self, g):
        return F.softplus(self.net(g))


class ESCTModel:
    def __init__(self, p_dim, emb_dim, device, seed=0, w1=True, dir_weight=1.0,
                 orth_weight=0.5, res_l2_weight=1.0, mmd_weight=5.0, aux_weight=1.0,
                 lr=3e-4, ema=0.999, use_res_gate=True, dist_loss="sinkhorn",
                 res_eps=0.01):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.p_dim, self.emb_dim, self.device = p_dim, emb_dim, device
        self.dir_weight = dir_weight
        self.orth_weight = orth_weight
        self.res_l2_weight = res_l2_weight
        self.mmd_weight = mmd_weight
        self.aux_weight = aux_weight
        self.ema_decay = ema
        self.use_res_gate = use_res_gate
        self.dist_loss = dist_loss
        self.res_eps = res_eps
        self.cell_enc = CellEnc(p_dim, 128).to(device)
        self.cond_enc = CondEnc(emb_dim).to(device)
        self.vel = ResidualVelocity(128, 32, p_dim).to(device)
        self.vel_ema = copy.deepcopy(self.vel)
        self.gamma = GammaGate(32).to(device)
        self.res_gate = ResGate(32).to(device) if use_res_gate else None
        self.aux_dec = AuxDec(128, p_dim).to(device)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.w1 = w1
        self.pot = W1Pot(p_dim, emb_dim).to(device) if w1 else None
        self.opt_pot = torch.optim.Adam(self.pot.parameters(), lr=1e-3) if w1 else None

    def parameters(self):
        params = (list(self.cell_enc.parameters()) + list(self.cond_enc.parameters()) +
                  list(self.vel.parameters()) + list(self.gamma.parameters()))
        if self.res_gate is not None:
            params += list(self.res_gate.parameters())
        return params + list(self.aux_dec.parameters())

    def g(self, e):
        return self.cond_enc(e)

    def ema_update(self):
        for p, pe in zip(self.vel.parameters(), self.vel_ema.parameters()):
            pe.data.mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay)

    def velocity(self, h, t, g, d_prior):
        if self.use_res_gate:
            r = self.res_gate(g) * self.vel(h, t, g)
        else:
            r = self.res_eps * self.vel(h, t, g)
        return self.gamma(g) * d_prior[None, :] + r

    def freeze_residual(self, freeze=True):
        for p in self.vel.parameters():
            p.requires_grad = not freeze
        self.opt = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad], lr=3e-4)

    def w1_dir(self, x0, e):
        xg = x0.detach().clone().requires_grad_(True)
        h = self.pot(xg, e.detach())
        grad = torch.autograd.grad(h.sum(), xg, retain_graph=False)[0].detach()
        return grad / (grad.norm(dim=1, keepdim=True) + 1e-6)

    def train_step(self, x0, x1, e, d_target=None, d_prior=None):
        g = self.g(e)
        delta = x1 - x0
        t = torch.rand(len(x0), 1, device=self.device)
        s = torch.clamp(t + 0.05, max=1.0)
        xt = x0 + t * delta
        xs = x0 + s * delta
        h0 = self.cell_enc(xt)
        hs = self.cell_enc(xs)
        vt = self.velocity(h0, t, g, d_prior)
        with torch.no_grad():
            if self.use_res_gate:
                vs = (self.gamma(g) * d_prior[None, :] +
                      self.res_gate(g) * self.vel_ema(hs, s, g))
            else:
                vs = (self.gamma(g) * d_prior[None, :] +
                      self.res_eps * self.vel_ema(hs, s, g))
            fs = xs + (1.0 - s) * vs
        ft = xt + (1.0 - t) * vt
        norm = delta.norm(dim=1, keepdim=True) + 1e-6
        loss = ((vt - delta) ** 2).mean() + \
               ((ft - fs) ** 2).mean() + ((vt - vs) ** 2).mean()
        if self.mmd_weight > 0:
            pred1 = x0 + vt
            if self.dist_loss == "mmd":
                loss = loss + self.mmd_weight * rbf_mmd(pred1, x1)
            else:
                loss = loss + self.mmd_weight * sinkhorn_loss(pred1, x1)
        if self.dir_weight > 0 and d_target is not None:
            w = norm.squeeze(1) / norm.sum()
            loss = loss - self.dir_weight * (
                (vt / (vt.norm(dim=1, keepdim=True) + 1e-6) * d_target[None, :]).sum(1) * w).mean()
        if self.orth_weight > 0 and d_prior is not None:
            r = self.vel(h0, t, g)
            rn = r / (r.norm(dim=1, keepdim=True) + 1e-6)
            loss = loss + self.orth_weight * ((rn * d_prior[None, :]).sum(1) ** 2).mean()
        if self.res_l2_weight > 0:
            loss = loss + self.res_l2_weight * self.vel(h0, t, g).pow(2).mean()
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

    def predict(self, x, e, d_prior):
        with torch.no_grad():
            g = self.cond_enc(torch.as_tensor(e, device=self.device, dtype=torch.float32)
                              .unsqueeze(0).expand(len(x), -1))
            xt = torch.as_tensor(x, device=self.device, dtype=torch.float32)
            h = self.cell_enc(xt)
            t = torch.zeros(len(x), 1, device=self.device)
            return xt + self.velocity(h, t, g, d_prior)
