import sys

import torch as th
import torch.nn as nn
import numpy as np

import runner.method as mtd


def get_schedule(args, config):
    if config['type'] == "quad":
        betas = (np.linspace(config['beta_start'] ** 0.5, config['beta_end'] ** 0.5, config['diffusion_step'], dtype=np.float64) ** 2)
    elif config['type'] == "linear":
        betas = np.linspace(config['beta_start'], config['beta_end'], config['diffusion_step'], dtype=np.float64)
    else:
        betas = None

    betas = th.from_numpy(betas).float()
    alphas = 1.0 - betas
    alphas_cump = alphas.cumprod(dim=0)

    return betas, alphas_cump


class Schedule(object):
    def __init__(self, args, config):
        device = th.device(args.gpu)
        betas, alphas_cump = get_schedule(args, config)

        self.betas, self.alphas_cump = betas.to(device), alphas_cump.to(device)
        self.alphas_cump_pre = th.cat([th.ones(1).to(device), self.alphas_cump[:-1]], dim=0)
        self.total_step = config['diffusion_step']

        self.method = mtd.choose_method(args.method)
        self.ets = None

    def diffusion(self, img, t_end, t_start=0, noise=None):
        if noise is None:
            noise = th.randn_like(img)
        alpha = self.alphas_cump.index_select(0, t_end).view(-1, 1, 1, 1)
        img_n = img * alpha.sqrt() + noise * (1 - alpha).sqrt()

        return img_n, noise

    def denoising(self, img_n, t_end, t_start, model, first_step=False):
        if first_step:
            self.ets = []
        img_next = self.method(img_n, t_start, t_end, model, self.alphas_cump, self.ets)

        return img_next

