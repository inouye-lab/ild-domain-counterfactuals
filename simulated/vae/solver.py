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
import geomloss
import wandb

from models import VAEAuto


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
        self.obs_dim = config.obs_dim
        self.lamb_kld = config.lamb_kld

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

        self.model = VAEAuto(self.config)

        self.f_opt = torch.optim.Adam(self.model.F.parameters(), self.lr_f, (self.beta1, self.beta2))
        self.g_opt = torch.optim.Adam(self.model.G.parameters(), self.lr_g, (self.beta1, self.beta2))

        self.model.to(self.device)

    def set_model_mode(self, set_to_train=True):
        self.model.train(set_to_train)

    def reset_grad(self):
        self.f_opt.zero_grad()
        self.g_opt.zero_grad()

    def recon_loss(self, x_hat, x_real):
        return torch.nn.functional.mse_loss(x_hat, x_real, reduction='none').sum(dim=1).mean()

    def latent_alignment_loss(self, mu, log_var):
        # clip log_var to avoid nan
        log_var = torch.clamp(log_var, max=15)  # max value of exp(15) is 3269017
        return torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim = 1), dim = 0)
    def train(self):

        data_loader = self.data_loader
        data_iter = iter(data_loader)

        start_iters = 0

        for i in tqdm(range(start_iters, self.num_iters), desc='Training'):

            self.set_model_mode(set_to_train=True)  # set all models to train mode

            self.iter = i # for logging
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
            #                         2. Forward and Losses                                       #
            # =================================================================================== #
            # First project to intermediate space
            z_from_G = self.model.G.x_to_z(x_real, d_real)
            # Use F_inv to recover the noise
            eps_hat, mu, log_var = self.model.F.z_to_eps(z_from_G, d_real, return_mu_logvar=True)
            # Now project back to original space
            z_from_F = self.model.F.eps_to_z(eps_hat, d_real)
            x_back = self.model.G.z_to_x(z_from_F, d_real)

            # Reconstruction loss
            loss_recon = self.recon_loss(x_real, x_back)
            #Alignment loss
            loss_alignment = self.latent_alignment_loss(mu, log_var)

            # =================================================================================== #
            #                                 3. ginv, w - spa                                    #
            # =================================================================================== #
            self.reset_grad()

            # total_loss =  loss_recon + 4 * loss_alignment
            total_loss =  loss_recon + self.lamb_kld * loss_alignment
            total_loss.backward()

            self.f_opt.step()
            self.g_opt.step()

            if self.wandb and (i + 1) % self.step_check == 0:
                wandb.log({'Train-loss/recon': loss_recon.item(),
                           'Train-loss/align': loss_alignment.item(),
                           'Train-loss/total': total_loss.item(),
                          }, step=i)

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #
            if (i + 1) % self.step_check == 0:
                self.set_model_mode(set_to_train=False)

                self.calc_alignment_error(idx=i, data_split='val')
                self.calc_cf_align_error(idx=i, data_split='test')

            # save the models
            if (i + 1) % self.step_save == 0:
                self.set_model_mode(set_to_train=False)
                save_dir = f'{self.save_dir}/{self.run_name}_{self.seed}'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(self.model.state_dict(), f'{save_dir}/ckpt_{i+1}.pt')

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
            # First project to intermediate space
            z_from_G = self.model.G.x_to_z(x_batch, d_batch)
            # Use F_inv to recover the noise
            eps_hat, mu, log_var = self.model.F.z_to_eps(z_from_G, d_batch, return_mu_logvar=True)
            # Now project back to original space
            z_from_F = self.model.F.eps_to_z(eps_hat, d_batch)
            x_back = self.model.G.z_to_x(z_from_F, d_batch)

            # Reconstruction loss
            loss_recon = self.recon_loss(x_batch, x_back)
            #Alignment loss
            loss_alignment = self.latent_alignment_loss(mu, log_var)
            total_loss = loss_recon + self.lamb_kld * loss_alignment
        if self.wandb:
            wandb.log({f'{data_split}-loss/total': total_loss.item(),
                       }, step=i)

    def calc_cf_align_error(self, idx=None, data_split=None, no_vis=False):
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

            domaindiff_counterfactual_loss = 0  # counterfactual loss for each domain pair (d != dp)
            domaindiff_id_loss = 0
            domaindiff_alignment_loss = 0

            alignloss_fn = geomloss.SamplesLoss(loss='sinkhorn')

            for d in self.domain_list:

                x_d_real = x_batch[d_batch==d]
                d_real = d_batch[d_batch==d]
                eps_real = eps_batch[d_batch ==d]

                for dp in self.domain_list:
                    # Computing the counterfactual loss
                    # generate ground truth counterfactuals
                    z_d_to_dp_gt = dataloader.dataset.noise_to_z(eps_real, dp)
                    x_d_to_dp_gt = dataloader.dataset.z_to_x(z_d_to_dp_gt)
                    # generate full estimated counterfactuals: from x_d --> z_d --> eps --> z_d_to_dp --> x_d_to_dp
                    eps_est = self.model.inverse(x_d_real, torch.ones_like(d_real)*d)
                    x_d_to_dp_est = self.model(eps_est, torch.ones_like(d_real)*dp)
                    # compute loss between the estimated and ground truth counterfactuals in x space
                    counterfactual_loss = torch.nn.functional.mse_loss(x_d_to_dp_est, x_d_to_dp_gt)

                    # compute alignment loss
                    x_dp_real = x_batch[d_batch==dp]
                    alignment_loss = alignloss_fn(x_d_to_dp_est, x_dp_real)

                    if d != dp:
                        domaindiff_counterfactual_loss += counterfactual_loss
                        domaindiff_id_loss += torch.nn.functional.mse_loss(x_d_real, x_d_to_dp_gt)
                        domaindiff_alignment_loss += alignment_loss

                    if (i + 1) % self.step_vis == 0 and self.wandb and not no_vis and d!=dp and d<=1 and dp<=1 :
                        # plotting full counterfactual distribution:
                        self.plot_cf_eval(x_d_to_dp_est, x_d_to_dp_gt, d, dp, i, title=f'{data_split}_X_cf')

            if self.wandb:
                n_ordered_pairs = len(list(combinations(self.domain_list, 2)))*2
                wandb.log({
                    f'{data_split}_counterfactual_error/d_neq_dp_avg': domaindiff_counterfactual_loss/n_ordered_pairs
                    },step=i)
                wandb.log({
                    f'{data_split}_id_error/d_neq_dp_avg': domaindiff_id_loss/n_ordered_pairs
                    },step=i)
                wandb.log({
                    f'{data_split}_alignment_error/d_neq_dp_avg': domaindiff_alignment_loss / n_ordered_pairs
                }, step=i)


    def plot_cf_eval(self, cf_est, cf_gt, d, dp, step, title='X', MAX_SAMPLES=1000, xlim=(0,1),ylim=(0,1)):
        # combine all z_d's into a single dataframe
        z_dfs = []
        cf_est = pd.DataFrame(cf_est.detach().cpu().numpy())
        cf_est['Data Type'] = 'Fake'
        z_dfs.append(cf_est)
        cf_gt = pd.DataFrame(cf_gt.detach().cpu().numpy())
        cf_gt['Data Type'] = 'True'
        z_dfs.append(cf_gt)
        z_df = pd.concat(z_dfs, axis=0).reset_index(drop=True)
        z_df = z_df.sample(min(MAX_SAMPLES, z_df.shape[0]), replace=False)  # subsample to MAX_SAMPLES
        img = sns.pairplot(z_df, hue='Data Type', hue_order=['True','Fake'],grid_kws={'diag_sharey': True},plot_kws={"s": 3},palette="husl")
        #img.set(xlim=xlim, ylim =ylim)
        wandb.log({f'VIS_DIST_{title}/{d}_to_{dp}': wandb.Image(img.figure)}, step=step)

        nd = len(z_df.columns) - 1
        ij = permutations(list(range(nd)),2)
        for i,j in ij:
            for idx in range(10):
                img.axes[i, j].arrow(cf_gt[j][idx], cf_gt[i][idx], cf_est[j][idx]-cf_gt[j][idx], cf_est[i][idx]-cf_gt[i][idx], width=0.01, color='black')
        #img.set(plot_kws={"s": 1})
        wandb.log({f'VIS_CF_{title}/{d}_to_{dp}': wandb.Image(img.figure)}, step=step)