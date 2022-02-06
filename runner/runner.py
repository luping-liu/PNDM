import os
import sys
import tqdm
import time
import torch as th
import torch.nn as nn
import torch.optim as optimi
import torch.utils.data as data
import torchvision.utils as tvu
import numpy as np

from dataset import get_dataset, inverse_data_transform
from model.ema import EMAHelper


def get_optim(params, config):
    if config['optimizer'] == 'adam':
        optim = optimi.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'],
                            betas=(config['beta1'], 0.999), amsgrad=config['amsgrad'],
                            eps=config['eps'])
    elif config['optimizer'] == 'sgd':
        optim = optimi.SGD(params, lr=config['lr'], momentum=0.9)
    else:
        optim = None

    return optim


class Runner(object):
    def __init__(self, args, config, schedule, model):
        self.args = args
        self.config = config
        self.diffusion_step = 1000
        self.device = th.device(args.gpu)

        self.schedule = schedule
        self.model = model

    def train(self):
        schedule = self.schedule
        model = self.model
        optim = get_optim(model.parameters(), self.config['Optim'])

        config = self.config['Dataset']
        dataset, test_dataset = get_dataset(self.args, config)
        train_loader = data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True,
                                       num_workers=config['num_workers'])

        config = self.config['Train']
        if config['ema']:
            ema = EMAHelper(mu=config['ema_rate'])
            ema.register(model)
        else:
            ema = None

        epoch, step = 0, 0
        for epoch in range(epoch, config['epoch']):
            for i, (img, y) in enumerate(train_loader):
                n = img.shape[0]
                model.train()
                step += 1
                t = th.randint(low=0, high=self.total_step, size=(n // 2 + 1,))
                t = th.cat([t, self.total_step - t - 1], dim=0)[:n].to(self.device)
                img = img.to(self.device)

                img_n, noise = schedule.diffusion(img, t)
                noise_p = model(img_n, t)

                if config['loss_type'] == 'linear':
                    loss = (noise_p - noise).abs().sum(dim=(1, 2, 3)).mean(dim=0)
                elif config['loss_type'] == 'square':
                    loss = (noise_p - noise).square().sum(dim=(1, 2, 3)).mean(dim=0)
                else:
                    loss = None

                optim.zero_grad()
                loss.backward()
                optim.step()
                if step % 50 == 0:
                    print(step, loss.item())

                if ema is not None:
                    ema.update(model)

    def sample(self):
        config = self.config['Sample']
        if config['mpi4py']:
            import mpi4py
        schedule = self.schedule
        model = self.model
        device = self.device

        model.load_state_dict(th.load(self.args.model_path), strict=True)
        model.eval()

        n = config['batch_size']
        total_num = config['total_num']

        skip = self.diffusion_step // self.args.sample_step
        seq = range(0, self.diffusion_step, skip)
        seq_next = [-1] + list(seq[:-1])
        image_num = 0

        config = self.config['Dataset']
        with th.no_grad():
            for _ in tqdm.tqdm(range(total_num // n + 1)):
                noise = th.randn(n, config['channels'], config['image_size'],
                                 config['image_size'], device=self.device)
                imgs = [noise]
                start = True

                for i, j in zip(reversed(seq), reversed(seq_next)):
                    t = (th.ones(n) * i).to(device)
                    t_next = (th.ones(n) * j).to(device)

                    img_t = imgs[-1].to(self.device)
                    img_next = schedule.denoising(img_t, t_next, t, model, start)
                    start = False

                    imgs.append(img_next.to('cpu'))

                img = imgs[-1]
                img = inverse_data_transform(config, img)
                for i in range(img.shape[0]):
                    if image_num+i > total_num:
                        break
                    tvu.save_image(img[i], os.path.join(self.args.image_path, f"{image_num+i}.png"))

                image_num += n

