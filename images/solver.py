from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torchvision.utils import make_grid

import wandb

import einops

from models import F_VAE_dense, F_VAE_relax_can, F_VAE_can
from models import GBetaVAE, GIndpendentBetaVAE
from models import KLDScheduler

import yaml

from tqdm.auto import tqdm
import subprocess

# TODO: Learning rate decay

class Solver(object):
    ''' Training and testing our CausalSpa'''

    def __init__(self,
                 train_loader,
                 val_loader,
                 test_loader,
                 config):
        print(f'Initiating RMNIST solver with config:\n{vars(config)}')

        # config
        self.config = config
        self.device = config.device
        self.seed = config.seed
        self.torch_generator = torch.Generator(device=self.device).manual_seed(self.seed)

        # data
        self.dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
        self.data_iters = {data_split: iter(dataloader) for data_split, dataloader in self.dataloaders.items()}
        self.domain_list = config.domain_list
        self.n_domains = len(self.domain_list)
        self.testing_batch_caches = {}
        self.dataset = config.dataset
        self.image_shape = config.image_shape

        # training hyperparameters
        self.num_iters = config.num_iters
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.lr_f = config.lr_f
        self.lr_g = config.lr_g

        self.use_weight_decay = config.use_weight_decay

        # training algorithm settings
        self.iter = -1
        self.kld_scheduler = KLDScheduler(config.kld_scheduler, 
                                          start_value=config.lamb_kld_start,
                                          end_value=config.lamb_kld_end,
                                          delay_iters=int(config.kld_step_delay_frac * config.num_iters),
                                          total_iters=int(config.kld_step_end_frac * config.num_iters))
        print(f'KLDScheduler: {self.kld_scheduler}, start_value: {self.kld_scheduler.start_value}, end_value: {self.kld_scheduler.end_value}, delay_iters: {self.kld_scheduler.delay_iters}, total_iters: {self.kld_scheduler.total_iters}')

        # model
        self.f_type = config.f_type
        self.k_spa = config.k_spa
        self.latent_dim = config.latent_dim

        # log and dir
        self.wandb = config.wandb
        self.step_validate = config.step_validate
        self.step_vis = config.step_vis
        self.step_save = config.step_save
        self.save_dir = config.save_dir
        self.save_final_dir = config.save_final_dir
        self.run_name = config.run_name

        self.build_model()


    def build_model(self):

        if self.f_type == 'dense':
            self.F = F_VAE_dense(self.config)
        elif self.f_type == 'relax_can':
            self.F = F_VAE_relax_can(self.config)
        elif self.f_type == 'can':
            self.F = F_VAE_can(self.config)

        if self.config.g_type == 'beta':
            self.G = GBetaVAE(self.config)
        elif self.config.g_type == 'independent':
            self.G = GIndpendentBetaVAE(self.config)

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
    
    def recon_loss(self, x_hat, x_real):
        return torch.nn.functional.mse_loss(x_hat, x_real, reduction='none').sum(dim=(1,2,3)).mean()

    def latent_alignment_loss(self, mu, log_var):
        # clip log_var to avoid nan
        log_var = torch.clamp(log_var, max=15)  # max value of exp(15) is 3269017
        return torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim = 1), dim = 0)

    def train(self):
        start_iters = 0
        best_val_loss = np.inf
        for i in tqdm(range(start_iters, self.num_iters), desc='Training'):

            self.set_model_mode(set_to_train=True)

            # =================================================================================== #
            #                         1. Preprocess input data                                    #
            # =================================================================================== #

            x_real, _, d_real = self._sample_dataloader('train')
            x_real = x_real.to(self.device)
            d_real = d_real.to(self.device)

            # =================================================================================== #
            #                         2. Forward and Losses                                       #
            # =================================================================================== #

            # First project to intermediate space
            z_from_G = self.G.x_to_z(x_real, d_real)
            # Use F_inv to recover the noise
            eps_hat, mu, log_var = self.F.z_to_eps(z_from_G, d_real, return_mu_logvar=True)
            # Now project back to original space
            z_from_F = self.F.eps_to_z(eps_hat, d_real)
            x_back = self.G.z_to_x(z_from_F, d_real)

            # Reconstruction loss
            loss_recon = self.recon_loss(x_real, x_back)
            #Alignment loss
            loss_alignment = self.latent_alignment_loss(mu, log_var)

            # =================================================================================== #
            #                         3. Step Models                                              #
            # =================================================================================== #
            self.reset_grad()

            # total_loss =  loss_recon + 4 * loss_alignment
            total_loss =  loss_recon + self.kld_scheduler.get_weight(step=True) * loss_alignment
            total_loss.backward()

            self.f_opt.step()
            self.g_opt.step()

            if self.wandb:
                wandb.log({'Train-loss/recon': loss_recon.item(),
                           'Train-loss/align': loss_alignment.item(),
                           'Train-loss/total': total_loss.item(),
                           'kld_weight': self.kld_scheduler.get_weight(step=False),
                          }, step=i)

            # =================================================================================== #
            #                         4. Validate, Visualize, Checkpoint                          #
            # =================================================================================== #
            self.set_model_mode(set_to_train=False)

            # log validation loss
            if (i + 1) % self.step_validate == 0:
                # Create counterfactual visualizations
                self.visualize_counterfactuals(i, data_split='val')
                # Compute validation loss
                val_loss_recon, val_loss_align = self.calc_validation_losses('val', normalize_by_batch_count=True)
                val_loss = val_loss_recon + val_loss_align * self.kld_scheduler.get_weight(step=True)
                if self.wandb:
                    wandb.log({'Val-loss/recon': val_loss_recon,
                                'Val-loss/align': val_loss_align,
                                'Val-loss/total': val_loss,
                                }, step=i)
                # cf_total_error, cf_d_eq_dp, cf_d_not_eq_dp = self.cal_cf_error('test')
                # if self.wandb:
                #     wandb.log({'Val-loss/recon': val_loss_recon,
                #                 'Val-loss/align': val_loss_align,
                #                'Test/CF-total':cf_total_error,
                #                'Test/CF-d_eq_dp':cf_d_eq_dp,
                #                'Test/CF-d_not_eq_dp':cf_d_not_eq_dp,
                #                 }, step=i)

            # save model checkpoints
            if (i + 1) % self.step_save == 0:
                save_dir = Path(self.save_dir) / f'{self.dataset}' / f'{self.run_name}'
                if not save_dir.exists():
                    save_dir.mkdir(parents=True)
                torch.save(self.G.state_dict(), str(save_dir / f'G_{i+1}.pt'))
                torch.save(self.F.state_dict(), str(save_dir / f'F_{i+1}.pt'))

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.G.state_dict(), str(save_dir / f'G_best.pt'))
                    torch.save(self.F.state_dict(), str(save_dir / f'F_best.pt'))

                # find best ckpt
                # save config file
                if not (save_dir / 'config.yml').exists():
                    with open(save_dir / 'config.yml', 'w') as f:
                        yaml.dump(self.config, f)

        # if self.save_final_dir is not None:
        #     source = Path(f'/local/scratch/a/zhou1059/ild_dg/images/{self.save_dir}') / f'{self.dataset}' / f'{self.run_name}'
        #     destination = Path(f'honeydew:/local/scratch/a/zhou1059/ild_dg/images/{self.save_final_dir}') / f'{self.dataset}'
        #     command = ["rsync", "--mkpath", "-r", source, destination]
        #     subprocess.run(command, check=True)

    def _sample_dataloader(self, data_split, put_on_device=False):
        try:
            x_batch, y_batch, d_batch = next(self.data_iters[data_split])
            if torch.all(d_batch == d_batch[0]):
                # this batch had all samples from the same domain, so skip it
                print(f'Encountered a batch with all samples from the same domain in {data_split} loader, ',
                       'skipping this batch...')
                return self._sample_dataloader(data_split, put_on_device=put_on_device)

        except StopIteration:
            # we've reached the end of the dataloader, so reset and get the first batch
            self.data_iters[data_split] = iter(self.dataloaders[data_split])
            return self._sample_dataloader(data_split, put_on_device=put_on_device)
        
        if put_on_device:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            d_batch = d_batch.to(self.device)
        return x_batch, y_batch, d_batch

    def _get_cached_batch(self, data_split):
            # load in data batch (we use a cache to reload the same batch for speed and so we have the same test every time)
            batch = self.testing_batch_caches.get(data_split)
            if batch is None:
                # this is the first run, so load the batches from a fresh dataloader and cache
                x_batch, y_batch, d_batch = next(iter(self.dataloaders[data_split]))
                eps_batch = torch.randn(x_batch.shape[0], self.latent_dim, generator=self.torch_generator, device=self.device)
                batch = (x_batch.to(self.device), y_batch.to(self.device), d_batch.to(self.device), eps_batch.to(self.device))
                self.testing_batch_caches[data_split] = batch
            return batch

    def calc_validation_losses(self, data_split, normalize_by_batch_count=False):
        """Calculate recon and alignment loss for `data_split` set"""
        assert data_split in ['val', 'test'], f'Invalid validation/test data split: {data_split}'
        self.set_model_mode(set_to_train=False)
        running_loss_recon = 0.0
        running_loss_align = 0.0
        with torch.no_grad():
            n_batches = 0
            for x_real, _, d_real in self.dataloaders[data_split]:
                x_real = x_real.to(self.device)
                d_real = d_real.to(self.device)    
                
                # First project to intermediate space
                z_from_G = self.G.x_to_z(x_real, d_real)
                # Use F_inv to recover the noise
                eps_hat, mu, log_var = self.F.z_to_eps(z_from_G, d_real, 
                                                       return_mu_logvar=True, set_epsilon_to_mean=True)
                # Now project back to original space
                z_from_F = self.F.eps_to_z(eps_hat, d_real)
                x_back = self.G.z_to_x(z_from_F, d_real)

                # Reconstruction loss
                running_loss_recon += self.recon_loss(x_real, x_back).item()
                #Alignment loss
                running_loss_align += self.latent_alignment_loss(mu, log_var).item()
                # Increment batch count
                n_batches += 1

        if normalize_by_batch_count:
            return running_loss_recon / n_batches, running_loss_align / n_batches
        else:
            return running_loss_recon, running_loss_align

    def cal_cf_error(self, data_split):
        assert data_split in ['test'], f'Invalid test data split: {data_split}'
        self.set_model_mode(set_to_train=False)
        running_cf_total_error = 0.0
        running_cf_d_eq_dp = 0.0
        running_cf_d_not_eq_dp = 0.0
        running_d_not_eq_dp_count = 0
        running_d_eq_dp_count = 0

        
        with torch.no_grad():
            for batch_idx, (x_real1, x_real2, _, d_real1, d_real2) in enumerate(self.dataloaders[data_split]):
                x_real1 = x_real1.to(self.device)
                x_real2 = x_real2.to(self.device)
                d_real1 = d_real1.to(self.device)
                d_real2 = d_real2.to(self.device)

                eps_back = self.F.z_to_eps(self.G.x_to_z(x_real1, d_real1), d_real1, set_epsilon_to_mean=True)
                counter = self.G.z_to_x(self.F.eps_to_z(eps_back, d_real2), d_real2)

                cf_error_per_sample = torch.mean(torch.nn.functional.mse_loss(counter, x_real2, reduction='none'), dim=[1, 2, 3])
                # Calculate *unweighted* error for d_real1 == d_real2 and d_real1 != d_real2
                cf_d_eq_dp_error = torch.sum(cf_error_per_sample[d_real1 == d_real2])
                cf_d_not_eq_dp_error = torch.sum(cf_error_per_sample[d_real1 != d_real2])
                running_cf_d_not_eq_dp += cf_d_not_eq_dp_error.item()
                running_cf_d_eq_dp += cf_d_eq_dp_error.item()
                running_d_eq_dp_count += torch.sum(d_real1 == d_real2)
                running_d_not_eq_dp_count += torch.sum(d_real1 != d_real2)
                # Calculate weighted total error
                cf_error_per_sample[d_real1 != d_real2] *= 1/self.n_domains
                running_cf_total_error += torch.sum(cf_error_per_sample).item()

        running_cf_d_not_eq_dp /= running_d_not_eq_dp_count  # take mean
        running_cf_d_eq_dp /= running_d_eq_dp_count  # take mean
        running_cf_total_error /= (running_d_not_eq_dp_count + running_d_eq_dp_count)  # take mean
        return running_cf_total_error, running_cf_d_eq_dp, running_cf_d_not_eq_dp

    def visualize_counterfactuals(self, idx=None, data_split=None, use_cache=False):
            if not self.wandb:
                # do nothing if wandb is not enabled
                return None

            i = idx if idx is not None else self.iter
            assert data_split in ['train', 'val', 'test'], f'This value must be train, test or val, got {data_split}'
            if use_cache:
                x_batch, y_batch, d_batch, eps_batch = self._get_cached_batch(data_split)
            else:
                x_batch, y_batch, d_batch = self._sample_dataloader(data_split)
                eps_batch = torch.randn(x_batch.shape[0], self.latent_dim,
                                        generator=self.torch_generator, device=self.device)

            # Splitting the batch such that we have one sample from each domain
            x_temp = []
            d_temp = []
            for domain in range(self.n_domains):
                assert (d_batch == domain).sum() > 0, f'No samples from domain {domain} in batch'
                x_temp.append(x_batch[d_batch == domain][0])
                d_temp.append(domain)
            x_temp = torch.stack(x_temp).to(self.device)
            d_temp = torch.tensor(d_temp).to(self.device)
            
            with torch.no_grad():
                # visualize counterfactuals
                vis_list = [x_temp]
                for dd in self.domain_list:
                    eps_back = self.F.z_to_eps( self.G.x_to_z(x_temp, d_temp), d_temp, set_epsilon_to_mean=True)
                    counter = self.G.z_to_x( self.F.eps_to_z(eps_back, torch.ones_like(d_temp)*dd),
                                             torch.ones_like(d_temp)*dd )
                    vis_list.append(counter)
                x_vis = torch.stack(vis_list)  # has shape: (n_domains, n_samples, 1, 28, 28)
                if self.image_shape[0] == 3:
                    grid_img = einops.rearrange(x_vis, 'd b c h w -> (d h) (b w) (c)').cpu().numpy()
                elif self.image_shape[0] == 1:
                    grid_img = einops.rearrange(x_vis, 'd b c h w -> (d h) (b w c)').cpu().numpy()
                grid_img = wandb.Image(grid_img)
                wandb.log({'img_cf': grid_img},step=i)

                # visualize generated samples
                eps_temp = eps_batch[:10]
                vis_list = []
                for dd in self.domain_list:
                    d_temp = torch.ones(10) * dd
                    d_temp = d_temp.to(torch.int64).to(self.device)
                    vis_list.append(self.G.z_to_x( self.F.eps_to_z(eps_temp,d_temp), d_temp ))
                x_vis = torch.stack(vis_list)  # has shape: (n_domains, n_samples, 1, 28, 28)
                if self.image_shape[0] == 3:
                    grid_img = einops.rearrange(x_vis, 'd b c h w -> (d h) (b w) (c)').cpu().numpy()
                elif self.image_shape[0] == 1:
                    grid_img = einops.rearrange(x_vis, 'd b c h w -> (d h) (b w c)').cpu().numpy()
                grid_img = wandb.Image(grid_img)
                wandb.log({'img_gen': grid_img},step=i)