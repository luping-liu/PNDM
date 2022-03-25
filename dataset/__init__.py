# MIT License
#
# Copyright (c) 2020 Jiaming Song
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import torch
import numbers
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.datasets import CIFAR10
from dataset.celeba import CelebA
# from dataset.ffhq import FFHQ
from dataset.lsun import LSUN
from torch.utils.data import Subset
import numpy as np


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


def get_dataset(args, config):
    if config['random_flip'] is False:
        tran_transform = test_transform = transforms.Compose(
            [transforms.Resize(config['image_size']), transforms.ToTensor()]
        )
    else:
        tran_transform = transforms.Compose(
            [
                transforms.Resize(config['image_size']),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.Resize(config['image_size']), transforms.ToTensor()]
        )

    if config['dataset'] == "CIFAR10":
        dataset = CIFAR10(
            os.path.join(os.getcwd(), "temp", "cifar10"),
            train=True,
            download=True,
            transform=tran_transform,
        )
        test_dataset = CIFAR10(
            os.path.join(os.getcwd(), "temp", "cifar10"),
            train=False,
            download=True,
            transform=test_transform,
        )

    elif config['dataset'] == "CELEBA":
        cx = 89
        cy = 121
        x1 = cy - 64
        x2 = cy + 64
        y1 = cx - 64
        y2 = cx + 64
        if config['random_flip']:
            dataset = CelebA(
                root=os.path.join(os.getcwd(), "temp", "celeba"),
                split="train",
                transform=transforms.Compose(
                    [
                        Crop(x1, x2, y1, y2),
                        transforms.Resize(config['image_size']),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]
                ),
                download=True,
            )
        else:
            dataset = CelebA(
                root=os.path.join(os.getcwd(), "temp", "celeba"),
                split="train",
                transform=transforms.Compose(
                    [
                        Crop(x1, x2, y1, y2),
                        transforms.Resize(config['image_size']),
                        transforms.ToTensor(),
                    ]
                ),
                download=True,
            )

        test_dataset = CelebA(
            root=os.path.join(os.getcwd(), "temp", "celeba"),
            split="test",
            transform=transforms.Compose(
                [
                    Crop(x1, x2, y1, y2),
                    transforms.Resize(config['image_size']),
                    transforms.ToTensor(),
                ]
            ),
            download=True,
        )

    elif config['dataset'] == "LSUN":
        train_folder = "{}_train".format(config['category'])
        val_folder = "{}_val".format(config['category'])
        if config['random_flip']:
            dataset = LSUN(
                root=os.path.join(os.getcwd(), "temp", "lsun"),
                classes=[train_folder],
                transform=transforms.Compose(
                    [
                        transforms.Resize(config['image_size']),
                        transforms.CenterCrop(config['image_size']),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToTensor(),
                    ]
                ),
            )
        else:
            dataset = LSUN(
                root=os.path.join(os.getcwd(), "temp", "lsun"),
                classes=[train_folder],
                transform=transforms.Compose(
                    [
                        transforms.Resize(config['image_size']),
                        transforms.CenterCrop(config['image_size']),
                        transforms.ToTensor(),
                    ]
                ),
            )

        test_dataset = LSUN(
            root=os.path.join(os.getcwd(), "temp", "lsun"),
            classes=[val_folder],
            transform=transforms.Compose(
                [
                    transforms.Resize(config['image_size']),
                    transforms.CenterCrop(config['image_size']),
                    transforms.ToTensor(),
                ]
            ),
        )
    #
    # elif config.data.dataset == "FFHQ":
    #     if config.data.random_flip:
    #         dataset = FFHQ(
    #             path=os.path.join(args.exp, "datasets", "FFHQ"),
    #             transform=transforms.Compose(
    #                 [transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor()]
    #             ),
    #             resolution=config.data.image_size,
    #         )
    #     else:
    #         dataset = FFHQ(
    #             path=os.path.join(args.exp, "datasets", "FFHQ"),
    #             transform=transforms.ToTensor(),
    #             resolution=config.data.image_size,
    #         )
    #
    #     num_items = len(dataset)
    #     indices = list(range(num_items))
    #     random_state = np.random.get_state()
    #     np.random.seed(2019)
    #     np.random.shuffle(indices)
    #     np.random.set_state(random_state)
    #     train_indices, test_indices = (
    #         indices[: int(num_items * 0.9)],
    #         indices[int(num_items * 0.9) :],
    #     )
    #     test_dataset = Subset(dataset, test_indices)
    #     dataset = Subset(dataset, train_indices)
    else:
        dataset, test_dataset = None, None

    return dataset, test_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config['uniform_dequantization']:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config['gaussian_dequantization']:
        X = X + torch.randn_like(X) * 0.01

    if config['rescaled']:
        X = 2 * X - 1.0
    elif config['logit_transform']:
        X = logit_transform(X)

    # if hasattr(config, "image_mean"):
    #     return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    # if hasattr(config, "image_mean"):
    #     X = X + config.image_mean.to(X.device)[None, ...]

    if config['logit_transform']:
        X = torch.sigmoid(X)
    elif config['rescaled']:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)
