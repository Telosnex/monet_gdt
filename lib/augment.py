"""Random resized crops for student distillation."""

import numpy as np
import torch


def crops(x, fl, g):
    """Random resized crops of x (B, 3, 384, 384), same size out; area 35-100 %, aspect
    3/4-4/3; the first half of the batch uncropped; fl = horizontal flip."""
    B = len(x)
    n = B // 2
    area = torch.empty(B).uniform_(0.35, 1.0, generator=g)
    ar = torch.exp(torch.empty(B).uniform_(np.log(3 / 4), np.log(4 / 3), generator=g))
    w = torch.sqrt(area * ar).clamp(max=1)
    h = torch.sqrt(area / ar).clamp(max=1)
    w[:n] = 1
    h[:n] = 1
    cx = (torch.rand(B, generator=g) * 2 - 1) * (1 - w)
    cy = (torch.rand(B, generator=g) * 2 - 1) * (1 - h)
    sx = torch.where(fl, -w, w)
    th = torch.zeros(B, 2, 3)
    th[:, 0, 0] = sx
    th[:, 0, 2] = cx
    th[:, 1, 1] = h
    th[:, 1, 2] = cy
    grid = torch.nn.functional.affine_grid(th.to(x.device), list(x.shape), align_corners=False)
    return torch.nn.functional.grid_sample(x, grid, mode='bicubic', padding_mode='border', align_corners=False)
