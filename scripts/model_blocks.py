"""Shared network blocks for ESCT."""

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


class W1Pot(nn.Module):
    def __init__(self, p_dim, emb_dim):
        super().__init__()
        self.f = LBNN(p_dim)
        self.g = nn.Sequential(nn.Linear(emb_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, e):
        return self.f(x) + self.g(e)
