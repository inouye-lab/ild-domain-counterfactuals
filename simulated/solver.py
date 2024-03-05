import os
import random
from itertools import combinations
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torchvision

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import permutations

import wandb

from models import F_dense, F_can, F_relax_can, G


# TODO: add predefined decreasing weight to sparsity loss

class Solver(object):
    ''' Training and testing our CausalSpa'''

    def __init__(self,
                 data_loader,
                 val_loader,
                 test_loader,
                 config):
        print(f'Initiating solver with config:\n{vars(config)}')

        # data
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = config.device
        self.domain_list = config.domain_list
        self.seed = config.seed
        self.testing_batch_caches = {}

        # training params
        self.num_iters = config.num_iters
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.lr_f = config.lr_f
        self.lr_g = config.lr_g

        # model
        self.f_type = config.f_type
        self.k_spa = config.k_spa
        self.latent_dim = config.latent_dim

        # log and dir
        self.wandb = config.wandb
        self.step_check = config.step_check
        self.step_vis = config.step_vis
        self.step_save = config.step_save
        self.save_dir = config.save_dir
        self.run_name = config.run_name

        self.iter = -1

        self.config = config

        self.build_model()

    def build_model(self):

        # true spa is soft cano
        if self.f_type == 'dense':
            self.F = F_dense(self.config)
        elif self.f_type == 'can':
            self.F = F_can(self.config)
        elif self.f_type == 'relax_can':
            self.F = F_relax_can(self.config)

        self.G = G(self.config)

        self.f_opt = torch.optim.Adam(self.F.parameters(), self.lr_f, (self.beta1, self.beta2))
        self.g_opt = torch.optim.Adam(self.G.parameters(), self.lr_g, (self.beta1, self.beta2))

        self.F.to(self.device)
        self.G.to(self.device)

    def set_model_mode(self, set_to_train=True):
        self.F.train(set_to_train)
        self.G.train(set_to_train)

    def reset_grad(self):
        self.f_opt.zero_grad()
        self.g_opt.zero_grad()

    def compute_loglikelihood(self, x, d):
        normal = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(self.latent_dim).to(self.device),
            torch.eye(self.latent_dim).to(self.device))
        z, logdet_x = self.G.inverse(x, return_jacobian=True)
        eps, logdet_z = self.F.inverse(z, d, return_jacobian=True)
        log_prob = normal.log_prob(eps) + logdet_z + logdet_x
        return torch.mean(log_prob)

    def train(self):

        data_loader = self.data_loader
        data_iter = iter(data_loader)

        start_iters = 0

        for i in tqdm(range(start_iters, self.num_iters), desc='Training'):

            self.set_model_mode(set_to_train=True)  # set all models to train mode

            self.iter = i  # for logging
            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            try:
                x_real, d_real, eps_real = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                x_real, d_real, eps_real = next(data_iter)

            x_real = x_real.to(self.device)
            d_real = d_real.to(self.device)
            # =================================================================================== #
            #                                 2. g, f - alignment                                 #
            # =================================================================================== #
            # MLE
            loss_gf = -self.compute_loglikelihood(x_real, d_real)

            # =================================================================================== #
            #                                 3. ginv, w - spa                                    #
            # =================================================================================== #
            self.reset_grad()
            loss_gf.backward()

            self.f_opt.step()
            self.g_opt.step()

            if self.wandb:
                wandb.log({'Loss/GF_align': loss_gf.item()}, step=i)

            # =================================================================================== #
            #                                 5. Miscellaneous                                    #
            # =================================================================================== #
            if (i + 1) % self.step_check == 0:
                self.calc_alignment_error(idx=i, data_split='val')
                self.calc_gt_counterfactual_error(idx=i, data_split='test')
                self.calc_alignment_error(idx=i, data_split='test')

            # save the models
            if (i + 1) % self.step_save == 0:
                save_dir = f'{self.save_dir}/{self.run_name}_{self.seed}'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(self.G.state_dict(), f'{save_dir}/G_{i + 1}.pt')
                torch.save(self.F.state_dict(), f'{save_dir}/F_{i + 1}.pt')

    def calc_alignment_error(self, idx=None, data_split=None):
        i = idx if idx is not None else self.iter
        assert data_split in ['train', 'val', 'test'], f'This value must be train, test or val, got {data_split}'
        if data_split == 'train':
            dataloader = self.data_loader
        elif data_split == 'val':
            dataloader = self.val_loader
        else:
            dataloader = self.test_loader
        with torch.no_grad():
            # get alignment loss for a batch of data from `data_split``
            batch = self.testing_batch_caches.get(data_split)
            if batch is None:
                # this is the first run, so load the batches and save
                x_batch, d_batch, eps_batch = next(iter(dataloader))
                batch = [x_batch.to(self.device), d_batch.to(self.device), eps_batch.to(self.device)]
                self.testing_batch_caches[data_split] = batch

            x_batch, d_batch, _ = batch
            alignment_loss = -self.compute_loglikelihood(x_batch, d_batch)
        if self.wandb:
            wandb.log({f'Loss/{data_split}_GF_align': alignment_loss}, step=i)
        return alignment_loss

    def calc_gt_counterfactual_error(self, idx=None, data_split=None, no_vis=False):
        i = idx if idx is not None else self.iter
        assert data_split in ['train', 'val', 'test'], f'This value must be train, test or val, got {data_split}'
        if data_split == 'train':
            dataloader = self.data_loader
        elif data_split == 'val':
            dataloader = self.val_loader
        else:
            dataloader = self.test_loader
        with torch.no_grad():
            # check the difference between generated counterfactuals and ground truth counterfactuals
            # load in data batch (we use a cache to reload the same batch for speed and so we have the same test every time)
            batch = self.testing_batch_caches.get(data_split)
            if batch is None:
                # this is the first run, so load the batches and save
                x_batch, d_batch, eps_batch = next(iter(dataloader))
                batch = [x_batch.to(self.device), d_batch.to(self.device), eps_batch.to(self.device)]
                self.testing_batch_caches[data_split] = batch
            x_batch, d_batch, eps_batch = batch

            if (i + 1) % self.step_vis == 0 and self.wandb and not no_vis:
                est_eps_batch = self.F.inverse(self.G.inverse(x_batch), d_batch)
                est_eps_batch = pd.DataFrame(est_eps_batch.cpu().numpy())
                img = sns.pairplot(est_eps_batch, grid_kws={'diag_sharey': True}, plot_kws={"s": 3})
                img.set(xlim=(-5, 5), ylim=(-5, 5))
                wandb.log({f'VIS_Gaussian': wandb.Image(img.figure)}, step=i)

            all_counterfactual_loss = 0  # go from x_d_real --> z_d_est --> eps_est --> z_d_to_dp_est --> x_d_to_dp_est
            domaindiff_counterfactual_loss = 0  # counterfactual loss for each domain pair (d != dp)
            domaindiff_id_loss = 0
            normalized_domaindiff_counterfactual_loss = 0  # the normalized counterfactual loss for each domain pair (d != dp)
            normalized_domaindiff_id_loss = 0

            for d in self.domain_list:

                x_d_real = x_batch[d_batch == d]
                d_real = d_batch[d_batch == d]
                eps_real = eps_batch[d_batch == d]

                for dp in self.domain_list:
                    # Computing the counterfactual loss
                    # generate ground truth counterfactuals
                    z_d_to_dp_gt = dataloader.dataset.noise_to_z(eps_real, dp)
                    x_d_to_dp_gt = dataloader.dataset.z_to_x(z_d_to_dp_gt)
                    # generate full estimated counterfactuals: from x_d --> z_d --> eps --> z_d_to_dp --> x_d_to_dp
                    z_d_est = self.G.inverse(x_d_real)
                    eps_est = self.F.inverse(z_d_est, torch.ones_like(d_real) * d)
                    z_d_to_dp_est = self.F(eps_est, torch.ones_like(d_real) * dp)
                    x_d_to_dp_est = self.G(z_d_to_dp_est)
                    # generate estimated counterfactuals with a ground truth z_d_to_dp
                    # x_d_to_dp_with_gt_z = self.G(z_d_to_dp_gt)
                    # compute loss between the estimated and ground truth counterfactuals in x space
                    counterfactual_loss = torch.nn.functional.mse_loss(x_d_to_dp_est, x_d_to_dp_gt)
                    # compute counterfactual loss with normalized x's
                    x_joint_mean = torch.mean(torch.concat([x_batch[d_batch == d], x_batch[d_batch == dp]], dim=0),
                                              dim=0)
                    x_joint_std = torch.std(torch.concat([x_batch[d_batch == d], x_batch[d_batch == dp]], dim=0), dim=0)
                    z_score_normalize = lambda x: (x - x_joint_mean) / x_joint_std
                    normalized_counterfactual_loss = torch.nn.functional.mse_loss(z_score_normalize(x_d_to_dp_est),
                                                                                  z_score_normalize(x_d_to_dp_gt))
                    # all_counterfactual_loss += counterfactual_loss

                    if d != dp:
                        domaindiff_counterfactual_loss += counterfactual_loss
                        normalized_domaindiff_counterfactual_loss += normalized_counterfactual_loss
                        domaindiff_id_loss += torch.nn.functional.mse_loss(x_d_real, x_d_to_dp_gt)
                        normalized_domaindiff_id_loss += torch.nn.functional.mse_loss(z_score_normalize(x_d_real),
                                                                                      z_score_normalize(x_d_to_dp_gt))

                    # if self.wandb:
                    #     wandb.log({f'{data_split}_counterfactual_error/d{d}_to_d{dp}': counterfactual_loss}, step=i)

                    if (i + 1) % self.step_vis == 0 and self.wandb and not no_vis and d != dp and d <= 1 and dp <= 1:
                        # plotting the counterfactual distributions for f, g, and full counterfactuals
                        # plotting f counterfactual distribution:
                        # self.plot_cf_eval(z_d_to_dp_est, z_d_to_dp_gt, d, dp, i, title=f'{data_split}_Z_df')
                        # plotting g counterfactual distribution:
                        # self.plot_cf_eval(x_d_to_dp_with_gt_z, x_d_to_dp_gt, d, dp, i, title=f'{data_split}_X_cf_given_Z_cf_gt')
                        # plotting full counterfactual distribution:
                        self.plot_cf_eval(x_d_to_dp_est, x_d_to_dp_gt, d, dp, i, title=f'{data_split}_X_cf')

            if self.wandb:
                n = len(self.domain_list)
                n_ordered_pairs = len(list(combinations(self.domain_list, 2))) * 2
                wandb.log({
                    f'{data_split}_counterfactual_error/d_neq_dp_avg': domaindiff_counterfactual_loss / n_ordered_pairs
                }, step=i)
                wandb.log({
                    f'{data_split}_id_error/d_neq_dp_avg': domaindiff_id_loss / n_ordered_pairs
                }, step=i)
                wandb.log({
                    f'{data_split}_normalized_counterfactual_error/d_neq_dp_avg': normalized_domaindiff_counterfactual_loss / n_ordered_pairs
                }, step=i)
                wandb.log({
                    f'{data_split}_normalized_id_error/d_neq_dp_avg': normalized_domaindiff_id_loss / n_ordered_pairs
                }, step=i)

    def plot_cf_eval(self, cf_est, cf_gt, d, dp, step, title='X', MAX_SAMPLES=1000, xlim=(0, 1), ylim=(0, 1)):
        # combine all z_d's into a single dataframe
        z_dfs = []
        cf_est = pd.DataFrame(cf_est.cpu().numpy())
        cf_est['Data Type'] = 'Fake'
        z_dfs.append(cf_est)
        cf_gt = pd.DataFrame(cf_gt.cpu().numpy())
        cf_gt['Data Type'] = 'True'
        z_dfs.append(cf_gt)
        z_df = pd.concat(z_dfs, axis=0).reset_index(drop=True)
        z_df = z_df.sample(min(MAX_SAMPLES, z_df.shape[0]), replace=False)  # subsample to MAX_SAMPLES
        img = sns.pairplot(z_df, hue='Data Type', hue_order=['True', 'Fake'], grid_kws={'diag_sharey': True},
                           plot_kws={"s": 3}, palette="husl")
        # img.set(xlim=xlim, ylim =ylim)
        wandb.log({f'VIS_DIST_{title}/{d}_to_{dp}': wandb.Image(img.figure)}, step=step)

        nd = len(z_df.columns) - 1
        ij = permutations(list(range(nd)), 2)
        for i, j in ij:
            for idx in range(10):
                img.axes[i, j].arrow(cf_gt[j][idx], cf_gt[i][idx], cf_est[j][idx] - cf_gt[j][idx],
                                     cf_est[i][idx] - cf_gt[i][idx], width=0.01, color='black')
        # img.set(plot_kws={"s": 1})
        wandb.log({f'VIS_CF_{title}/{d}_to_{dp}': wandb.Image(img.figure)}, step=step)