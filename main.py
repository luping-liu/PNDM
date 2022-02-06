import configparser
import argparse
import yaml
import sys
import os
import numpy as np
import torch as th

from runner.schedule import Schedule
from runner.runner import Runner
from model.ddim import Model


def args_and_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--runner", type=str, default='train',
                        help="Choose the mode of runner")
    parser.add_argument("--config", type=str, default='dev.yml',
                        help="Choose the config file")
    parser.add_argument("--method", type=str, default='F-PNDM',
                        help="Choose the numerical methods (DDIM, FON, S-PNDM, F-PNDM)")
    parser.add_argument("--sample_step", type=int, default=50,
                        help="Choose the total generation step")
    parser.add_argument("--gpu", type=str, default='cuda:3',
                        help="Choose the gpu to use")
    parser.add_argument("--image_path", type=str, default='temp/results',
                        help="Choose the path to save images")
    parser.add_argument("--model_path", type=str, default='temp/models/ddim/ema_cifar10.ckpt',
                        help="Choose the path of model")

    args = parser.parse_args()

    work_dir = os.getcwd()
    with open(f'{work_dir}/config/{args.config}', 'r') as f:
        config = yaml.safe_load(f)

    return args, config


def check_config():
    # image_size, total_step
    pass


if __name__ == "__main__":
    args, config = args_and_config()

    device = th.device(args.gpu)
    schedule = Schedule(args, config['Schedule'])
    model = Model(args, config['Model']).to(device)

    runner = Runner(args, config, schedule, model)
    if args.runner == 'train':
        runner.train()
    elif args.runner == 'sample':
        runner.sample()

