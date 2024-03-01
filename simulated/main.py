import os, sys, time, warnings
import argparse
import random
import numpy as np
import yaml

import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data_utils

from dataset import SimulatedDataset
from solver import Solver

import wandb

# import user variables for wandb

# parameters that might need to be changed when switching SCM
# d_embedding_dim: should be reasonably low
# latent_dim: should be the same as data dimension
# k_spa: change according to SCM

def main():
    parser = argparse.ArgumentParser(description='Train CausalSpa')

    # basics
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', default=0,
                        help='random seed (default: 0)')  # set to None for a random seed
    parser.add_argument('--pre_para_idx', type=int, default=None)

    # data
    parser.add_argument('--num_samples', type=int, default=100000, help='number of samples from each domain')
    parser.add_argument('--n_domains', type=int, default=2)
    parser.add_argument('--latent_dim', type=int, default=4)
    parser.add_argument('--int_set',type=str,default='2,3')

    # training params
    parser.add_argument('--lr_g', type=float, default=0.001)
    parser.add_argument('--lr_f', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--num_iters', type=int, default=50_000)

    # model params
    parser.add_argument('--f_type', type=str, default='relax_can')
    parser.add_argument('--k_spa', type=int, default=2)
    parser.add_argument('--g_init', type=str, default='id')
    parser.add_argument('--no_model_bias', action='store_true', default=False)
    parser.add_argument('--no_leaky_relu', action='store_true', default=False)
    parser.add_argument('--no_model_s', action='store_true', default=False)
    parser.add_argument('--leaky_relu_slope', type=float, default=0.5)

    # log and dir
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--save_dir', default='./saved')
    parser.add_argument('--no_wandb', action='store_true', default=False)
    parser.add_argument('--step_check', type=int, default=100)
    parser.add_argument('--step_vis', type=int, default=50_000)
    parser.add_argument('--step_save', type=int, default=1000000)
    parser.add_argument('--note', default='')
    parser.add_argument('--project_prefix', default='')

    # sweep (tells us whether this is being called as part of WandB sweep or not)
    parser.add_argument('--sweep', action='store_true', default=False)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    seed = args.seed
    if str(seed).lower() in ['none', 'false', 'rand', 'random']:
        seed = np.random.randint(1000)
        args.seed = seed
    else:
        args.seed = int(seed)

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    kwargs = {'num_workers': 1, 'pin_memory': False} if args.cuda else {}

    # ======================== #
    #         params           #
    # ======================== #
    args.int_set = [int(item) for item in args.int_set.split(',')]

    # ======================== #
    #         log              #
    # ======================== #
    args.wandb = not args.no_wandb
    args.run_name = f'[Domain{args.n_domains}-Dim{args.latent_dim}-Int{args.int_set}-{args.f_type}-{args.note}'
    # print(f'Run: {args.run_name} with args:\n{vars(args)}')
    if args.wandb or args.sweep:
        wandb.init(project=f"ICLR24-Simulated" if not args.sweep else None,
                   entity="" if not args.sweep else None,
                   name=args.run_name,
                   config=vars(args))
        wandb.run.log_code()
        wandb.define_metric('cf_full_test/domaindiff_avg', summary='min')
        wandb.define_metric('cf_full_val/domaindiff_avg', summary='min')

    # ======================== #
    #         data             #
    # ======================== #
    train_set = SimulatedDataset(num_samples=args.num_samples, n_domains=args.n_domains,
                                 latent_dim=args.latent_dim, int_set=args.int_set,
                                 model_seed=args.seed, noise_seed=args.seed + 100,
                                 device=args.device, relu_slope=args.leaky_relu_slope)
    train_loader = data_utils.DataLoader(train_set,
                                         batch_size=args.batch_size,
                                         shuffle=True, **kwargs)

    val_set = SimulatedDataset(num_samples=int(1000), n_domains=args.n_domains,
                               latent_dim=args.latent_dim, int_set=args.int_set,
                               model_seed=args.seed, noise_seed=args.seed + 101,
                               device=args.device, relu_slope=args.leaky_relu_slope,
                               normalization_latent_dict=train_set.normalization_latent_dict)
    val_loader = data_utils.DataLoader(val_set,
                                       batch_size=val_set.__len__(),
                                       shuffle=True, **kwargs)

    test_set = SimulatedDataset(num_samples=int(1000), n_domains=args.n_domains,
                                latent_dim=args.latent_dim, int_set=args.int_set,
                                model_seed=args.seed, noise_seed=args.seed + 102,
                                device=args.device, relu_slope=args.leaky_relu_slope,
                                normalization_latent_dict=train_set.normalization_latent_dict)
    test_loader = data_utils.DataLoader(test_set,
                                        batch_size=test_set.__len__(),
                                        shuffle=True, **kwargs)
    args.domain_list = train_set.domain_list
    args.n_domains = len(args.domain_list)

    # data and model

    # ======================== #
    #         training         #
    # ======================== #
    solver = Solver(train_loader, val_loader, test_loader, args)
    solver.train()



if __name__ == "__main__":
    main()