import os, sys, time, warnings
import argparse
import random
import numpy as np


import torch
import torch.nn as nn
import torchvision

from dataset import MnistColorRotated, MnistRotated
from solver import Solver

import wandb

KLD_SCHEDULER_PRESETS = {
    1: {'kld_scheduler': 'linear', 'lamb_kld_start': 0.001, 'lamb_kld_end': 1e5,
         'kld_step_delay_frac': 0.2, 'kld_step_end_frac': 1},

    2: {'kld_scheduler': 'linear', 'lamb_kld_start': 100, 'lamb_kld_end': 1e5,
         'kld_step_delay_frac': 0.5, 'kld_step_end_frac': 1},

    3: {'kld_scheduler': 'linear', 'lamb_kld_start': 1, 'lamb_kld_end': 1e-5,
         'kld_step_delay_frac': 0.3, 'kld_step_end_frac': 1},

    4: {'kld_scheduler': 'linear', 'lamb_kld_start': 1, 'lamb_kld_end': 1e10,
         'kld_step_delay_frac': 0, 'kld_step_end_frac': 1},

    5: {'kld_scheduler': 'linear', 'lamb_kld_start': 0.01, 'lamb_kld_end': 1e4,
         'kld_step_delay_frac': 0, 'kld_step_end_frac': 1},

    6: {'kld_scheduler': 'linear', 'lamb_kld_start': 0.01, 'lamb_kld_end': 1e4,
         'kld_step_delay_frac': 0.1, 'kld_step_end_frac': 1},
}

MODEL_PRESETS = {
    1: {'f_type': 'relax_can', 'g_type': 'beta'},
    2: {'f_type': 'can', 'g_type': 'beta'},
    3: {'f_type': 'dense', 'g_type': 'beta'},
    4: {'f_type': 'dense', 'g_type': 'independent'},
}

MODEL_PRESET_TO_NAME = {
    1: 'relax_can',
    2: 'can',
    3: 'dense',
    4: 'independent',
}

# import user variables for wandb
try:
    sys.path.append('..')
    sys.path.append('.')
    from wandb_env import WANDB_PROJECT_PREFIX
except ImportError:
    print('No wandb environment file found.\nTo make one, go to the above directory and',
          'create a file called `wandb_env.py` and enter: `WANDB_PROJECT_PREFIX = \'<your initials>\.',
          '\ne.g., sean\'s `wandb_env.py` looks like: `WANDB_PROJECT_PREFIX = \'SK\'.')
    WANDB_PROJECT_PREFIX = '00'
    print('Defaulting to wandb project prefix: ', WANDB_PROJECT_PREFIX)




def main():

    parser = argparse.ArgumentParser(description='Train CausalSpa')

    # ------------------- basics --------------------------- #
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')  # set to None for a random seed

    # ------------------- dataset params ------------------- #
    # parser.add_argument('--list_train_domains', '--list', nargs='+', default=['0','90'],
    #                    help='domains used during training')
    parser.add_argument('--dataset', type=str, default='rmnist')
    parser.add_argument('--subsample', action='store_true', default=False,help='subsample the dataset to be 1/10')

    # ------------------- training params ------------------- #
    parser.add_argument('--lr_g', type=float, default=1e-4)
    parser.add_argument('--lr_f', type=float, default=0.001)
    parser.add_argument('--kld_scheduler', type=str, default='none',  # if none, lamb_kld := lamb_kld_start
                        choices=['none', 'linear'])
    parser.add_argument('--kld_step_delay_frac', type=int, default=0.2)  # the percentage of total steps before scheduler starts
    parser.add_argument('--kld_step_end_frac', type=int, default=1)  # the percentage of total steps when the scheduler should end)
    parser.add_argument('--lamb_kld_start', type=float, default=1)  # this is the kld lambda at the start of training
    parser.add_argument('--lamb_kld_end', type=float, default=1)
    parser.add_argument('--ib_type', type=str, default='all', choices=['all', 'first', 'last'])
    parser.add_argument('--lamb_ib',type=float,default=0.0)

    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--num_iters', type=int, default=20_000)

    # sweep stuff
    parser.add_argument('--kld_preset_value', type=int, default=-1)
    parser.add_argument('--model_preset', type=int, default=-1)
    parser.add_argument('--sweep', action='store_true', default=False, help='whether this is a wandb sweep')

    # algorithm

    # model params
    parser.add_argument('--f_type', type=str, default='dense')
    parser.add_argument('--g_type', type=str, default='beta', help='type of generator', 
                        choices=['beta', 'independent'])
    #parser.add_argument('--inter_dim', type=int, default=25, help='dimension of intermediate layer of models')
    parser.add_argument('--latent_dim', type=int, default=10, help='size of latent space (i.e. dim of z)')
    parser.add_argument('--k_spa', type=int, default=2)
    parser.add_argument('--use_weight_decay', default=False, action='store_true')

    # log and dir
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--save_dir', default='saved/test')
    parser.add_argument('--save_final_dir',type=str, default=None)
    parser.add_argument('--no_wandb', action='store_true', default=False)
    parser.add_argument('--step_validate', type=int, default=500)  # check the validation loss every this many steps
    parser.add_argument('--step_vis', type=int, default=5_000)  # plot rmnist generated samples and counterfactuals
    parser.add_argument('--step_save', type=int, default=5_00000)  # save model every this many steps
    parser.add_argument('--note', default='')
    parser.add_argument('--project_prefix', default='')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # ======================== #
    #         randomness       #
    # ======================== #

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
    #TODO: check if num workers should be increased.

    # ======================== #
    #         misc check       #
    # ======================== #


    # ======================== #
    #         params           #
    # ======================== #
    if args.kld_preset_value > 0:
        kld_preset = KLD_SCHEDULER_PRESETS[args.kld_preset_value]
        print(f'Using KLD scheduler preset {kld_preset}')
        for k, v in kld_preset.items():
            setattr(args, k, v)  # set the values for the kld scheduler
    if args.model_preset > 0:
        model_preset = MODEL_PRESETS[args.model_preset]
        print(f'Using model preset {model_preset}')
        for k, v in model_preset.items():
            setattr(args, k, v)  # set the values for the model params

    # ======================== #
    #         log              #
    # ======================== #
    args.wandb = not args.no_wandb
    run_name = f'M{args.latent_dim}_k{args.k_spa}_beta{args.lamb_kld_start}'
    if args.lamb_ib > 0:
        run_name += f'_ib{args.ib_type}{args.lamb_ib}'
    run_name += f'_seed{args.seed}'
    if args.model_preset > 0:
        args.run_name = f'[{MODEL_PRESET_TO_NAME[args.model_preset]}]_{run_name}'
    else:
        args.run_name = f'[{args.f_type}]_{run_name}'

    args.project_prefix = WANDB_PROJECT_PREFIX if args.project_prefix == '' else args.project_prefix
    # print(f'Run: {args.run_name} with args:\n{vars(args)}')
    if args.wandb or args.sweep:
        wandb.init(project=f"[{args.project_prefix}]-colorrmnist" if not args.sweep else None,
                   entity="inouye-lab" if not args.sweep else None,
                   name=args.run_name,
                   config=vars(args))
        wandb.run.log_code()
        wandb.define_metric('Test/CF-total', summary='min')
        # wandb.define_metric('Val-loss/recon', summary='min')
        # wandb.define_metric('Val-loss/align', summary='min')

    # ======================== #
    #         data             #
    # ======================== #
    print('Loading data...')
    if args.dataset in ['rmnist', 'rfmnist', 'rscmnist']:
        args.image_shape = (1, 28, 28)
        args.list_train_domains = ['0', '15', '30', '45', '60']
        all_data = MnistRotated(args.list_train_domains, args.data_dir, train=True,
                                mnist_type=args.dataset,subsample=args.subsample)
        test_set = MnistRotated(args.list_train_domains, args.data_dir, train=False,
                                mnist_type=args.dataset, subsample=args.subsample)
    elif args.dataset == 'crmnist':
        args.image_shape = (3, 28, 28)
        args.list_train_domains = ['0', '90']
        all_data = MnistColorRotated(args.list_train_domains, args.data_dir, train=True)
        test_set = MnistColorRotated(args.list_train_domains, args.data_dir, train=False)

    train_size = int(0.9 * len(all_data))
    val_size = len(all_data) -  train_size
    train_set, val_set= torch.utils.data.random_split(all_data, [train_size, val_size],
                                                      generator=torch.Generator().manual_seed(args.seed))
    train_loader = torch.utils.data.DataLoader(train_set,
                                         batch_size=args.batch_size,
                                         shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_set,
                                       batch_size=args.batch_size,
                                       shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set,
                                        batch_size=args.batch_size,
                                        shuffle=False, **kwargs)


   # TODO: check if there are any problems with splitting the data this way.

    args.n_domains = len(args.list_train_domains)
    args.domain_list = list(range(len(args.list_train_domains)))
    print(f'Dataset sizes:\n\ttrain:{all_data.__len__()}, val:{val_set.__len__()}, test:{test_set.__len__()}')

    # ======================== #
    #         training         #
    # ======================== #
    solver = Solver(train_loader, val_loader, test_loader, args)
    solver.train()

# def check_if_spa_iters_is_disabled(args: argparse.Namespace) -> bool:
#     if args.spa_iters is None or (args.spa_iters is not None and args.num_iters < args.spa_iters):
#         print('!! Warning: Sparsity Loss is disabled !!')
#         return None
#     else:
#         return args.spa_iters


if __name__ == "__main__":
    main()