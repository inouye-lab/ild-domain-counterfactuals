import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
F = functional

import numpy as np


class KLDScheduler:
    """Scheduler for changing the KLD loss weight. Can be constant or linearly changing."""
    def __init__(self, scheduler_type, start_value, end_value=None,
                 total_iters=None, delay_iters=0):
        self.scheduler_type = str(scheduler_type).lower()
        assert self.scheduler_type in ['none', 'linear'], 'Invalid scheduler type'

        self.delay_iters = delay_iters
        self.total_iters = total_iters
        self.start_value = start_value  
        self.end_value = end_value

        self.step_idx = 0
        self.lamb_kld = start_value
        if self.scheduler_type == 'linear':
            self.increase_factor = (end_value - start_value) / total_iters

    def get_weight(self, step=False):
        out = self.lamb_kld
        if step:
            self.step()
        return out
    
    def step(self):
        self.step_idx += 1
        if self.scheduler_type == 'none':
            return self.lamb_kld
        elif self.scheduler_type == 'linear':
            if self.delay_iters < self.step_idx and self.step_idx < self.total_iters:
                if self.step_idx == self.delay_iters + 1:
                    self.lamb_kld = self.start_value
                else:
                    # increase KLD loss weight
                    self.lamb_kld += self.increase_factor
        else:
            raise ValueError('Invalid scheduler type')

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class DomainClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 4, 1, 0),          # B,  32, 25, 25
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 1, 0),          # B,  32, 22, 22
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 0),          # B,  64,  10, 10
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 0),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256*1*1)),                 # B, 256
            nn.Linear(256, self.n_classes),             # B, latent_dim
            nn.Softmax(-1)
        )

    def forward(self, x):
        return self.model(x)

    
class GBetaVAE(nn.Module):
    """A beta VAE model based on:
    https://github.com/1Konny/Beta-VAE/blob/master/solver.py
    """

    def __init__(self, config):
        super().__init__()
        self.latent_dim = config.latent_dim
        self.device = config.device
        self.config = config

        self.encoder = nn.Sequential(
            nn.Conv2d(self.config.image_shape[0], 32, 4, 2, 0),          # B,  32, 47, 47
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  10, 10
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 0),          # B,  64,  4,  4
            nn.ReLU(True),
            View((-1, 64*2*2)),                 # B, 256
            nn.Linear(64*2*2, self.latent_dim),             # B, latent_dim
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 64*2*2),               # B, 256
            View((-1, 64, 2, 2)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 0), # B,  64,  10,  10
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 22, 22
            nn.ReLU(True),
            nn.ConvTranspose2d(32, self.config.image_shape[0], 6, 2, 0),  # B, nc, 96, 96
            nn.Sigmoid()
        )

    def x_to_z(self, x, d=None):

        return self.encoder(x)

    def z_to_x(self, z, d=None):
        return self.decoder(z)

class GIndpendentBetaVAE(nn.Module):
    """A beta VAE model based on which is independent for each domain"""
    def __init__(self, config) -> None:
        super().__init__()
        self.latent_dim = config.latent_dim
        self.device = config.device
        self.config = config

        self.encoder_dict = nn.ModuleDict({
            str(d_idx):  nn.Sequential(
                nn.Conv2d(self.config.image_shape[0], 32, 4, 2, 0),  # B,  32, 47, 47
                nn.ReLU(True),
                nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  10, 10
                nn.ReLU(True),
                nn.Conv2d(64, 64, 4, 2, 0),  # B,  64,  4,  4
                nn.ReLU(True),
                View((-1, 64 * 2 * 2)),  # B, 256
                nn.Linear(64 * 2 * 2, self.latent_dim),  # B, latent_dim
                    ) for d_idx in range(self.config.n_domains)
        })
        self.decoder_dict = nn.ModuleDict({
            str(d_idx): nn.Sequential(
                nn.Linear(self.latent_dim, 64 * 2 * 2),  # B, 256
                View((-1, 64, 2, 2)),  # B, 256,  1,  1
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 64, 4, 2, 0),  # B,  64,  10,  10
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  32, 22, 22
                nn.ReLU(True),
                nn.ConvTranspose2d(32, self.config.image_shape[0], 6, 2, 0),  # B, nc, 96, 96
                nn.Sigmoid()
                ) for d_idx in range(self.config.n_domains)
        })

    def x_to_z(self, x, d):
        batch_size = x.shape[0]
        out = torch.zeros((batch_size, self.latent_dim), device=self.device)
        for d_idx in range(self.config.n_domains):
            out[d==d_idx] = self.encoder_dict[str(d_idx)](x[d==d_idx])
        return out

    def z_to_x(self, z, d):
        batch_size = z.shape[0]
        out = torch.zeros((batch_size, *self.config.image_shape), device=self.device)
        for d_idx in range(self.config.n_domains):
            out[d==d_idx] = self.decoder_dict[str(d_idx)](z[d==d_idx])
        return out


class F_VAE_can(nn.Module):
    def __init__(self, config):
        """
        Canonical autoregressive model with sparse dependence on domain and no interdependencies between the 
        domain invariant nodes (although the domain invariant nodes can *cause* the domain specific nodes).
        By construction, when d=0 we set F_d to the identity matrix (L_d=0, S_d=Id).
    
        Z_d = L_d @ Z_d + S_d @ epsilon
        L_d = Lower lower triangular matrix
        S_d = Diagonal scaling matrix

        F_d = (I - L_d)^{-1} @ S_d
        Z_d = F_d @ epsilon

        Example:
        k = 2, d = 2
        L_d = [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [a, b, 0, 0],
               [c, d, e, 0]]

        S_d = [[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, f, 0],
               [0, 0, 0, g]]
        """
        super().__init__()
        self.device = config.device
        self.n_domains = config.n_domains
        self.latent_dim = config.latent_dim
        self.generator = torch.Generator(device=self.device).manual_seed(config.seed)
        # setting up projection from dim to S, L, and bias
        self._setup_models(config)

    def _setup_models(self, config):
        """Setting up the models for the F_d_mu, F_d_sigma, and F_d_decode 
        (where each F_d_* has a L_d lower lower triangular matrix, S_d diagonal matrix, and bias)"""
        assert self.latent_dim > 0 and config.k_spa > 0, f'Must have N>=0 and k>=0, got N={self.latent_dim} and k={config.k_spa}'
        assert self.latent_dim > config.k_spa, f'Must have N > k, got N={self.latent_dim} and k={config.k_spa}'
        # getting the indices of the lower lower triangular matrix starting at row N-k
        # these indicies are the same for all F_d_*
        self.L_nonshared_idxs = torch.tensor(
            sum([[(i,j) for j in range(i)] for i in range(self.latent_dim-config.k_spa, self.latent_dim)], [])
        )
        # getting the indices of the diagonal matrix starting at row N-k
        self.S_nonshared_idxs = torch.tensor([[i, i] for i in range(self.latent_dim-config.k_spa, self.latent_dim)])
        self.S_shared_idxs = torch.tensor([[i, i] for i in range(self.latent_dim-config.k_spa)])
        # Building the F_d_mu, F_d_sigma, and F_d_decode embeddings
        self.d_to_L_and_S_nonshared_embeddings_dict = nn.ModuleDict({
            model_type: nn.ModuleDict(
                {'L': nn.Embedding(self.n_domains, self.L_nonshared_idxs.shape[0]),
                 'S': nn.Embedding(self.n_domains, self.S_nonshared_idxs.shape[0])})
                    for model_type in ['mu', 'sigma', 'decode']
        })
        # setting up bias terms (these have to be in two separate dicts since one is a parameter and one is a module)
        self.d_to_nonshared_bias_dict = nn.ModuleDict({  # not shared across domains
            model_type: nn.Embedding(self.n_domains, config.k_spa) 
                for model_type in ['mu', 'sigma', 'decode']
        })
        self.d_to_shared_bias_dict = nn.ParameterDict({  # shared across domains
            model_type: torch.nn.parameter.Parameter(torch.randn(self.latent_dim-config.k_spa))
                for model_type in ['mu', 'sigma', 'decode']
        })
        return None

    def eps_to_z(self, epsilon, d):
        """ Decoder: takes in epsilon and d, and generates z_d where
        f(epsilon, d) = F_d_decode @ epsilon + b_d_decode = z_d"""
        assert epsilon.ndim == 2, f'Epsilon must have shape: batch_dim, latent_dim'
        assert d.ndim == 1, f'd must have shape: batch_dim'
        # Find L_d and S_d, and use these to calculate F_d
        F_matrix = self._make_F(d, model_type='decode')
        # Calculate z_d
        z = (F_matrix @ epsilon.unsqueeze(-1)).squeeze() + self._make_bias(d, model_type='decode')
        return z

    def z_to_eps(self, z, d, return_mu_logvar=False, set_epsilon_to_mean=False):
        """ Encoder: Take in z and d, and generate mu_z and sigma_z, then generate epsilon
         If `set_epsilon_to_mean` is `False` then epsilon ~ N(mu_z, sigma_z), else epsilon = mu_z"""
        assert z.ndim == 2, f'z must have shape: batch_dim, latent_dim'
        assert d.ndim == 1, f'd must have shape: batch_dim'
        # Make the F_d matrices
        F_mu = self._make_F(d, model_type='mu')
        F_sigma = self._make_F(d, model_type='sigma')
        # Calculate mu_z and sigma_z
        mu_z = (F_mu @ z.unsqueeze(-1)).squeeze() + self._make_bias(d, model_type='mu')
        sigma_z = (F_sigma @ z.unsqueeze(-1)).squeeze() + self._make_bias(d, model_type='sigma')
        # Generate epsilon
        if set_epsilon_to_mean:
            epsilon = mu_z
        else:
            # Reparameterization trick
            epsilon = sigma_z * torch.randn(sigma_z.size(), generator=self.generator, device=self.device) + mu_z
        if return_mu_logvar:
            return epsilon, mu_z, sigma_z
        else:
            return epsilon
        
    def _make_bias(self, d, model_type):
        batch_size = d.shape[0]
        return torch.cat([self.d_to_shared_bias_dict[model_type].repeat(batch_size, 1),
                          self.d_to_nonshared_bias_dict[model_type](d)], dim=-1)

    def _make_F(self, d, model_type):
        """Returns F_d for the given model_type"""
        batch_size = d.shape[0]
        # Get the L_d and S_d matrices for all domains
        L_matrix, S_matrix = self._make_all_L_and_S(model_type)
        F_ds = torch.linalg.solve(
            torch.eye(self.latent_dim).unsqueeze(0).tile((self.n_domains, 1, 1)).to(self.device) - L_matrix, S_matrix
            )  # a more numerically stable way of doing (I - L_d)^{-1} @ S_d
        # Select the correct F_d for each d in the batch
        return torch.nn.functional.embedding(
                d, F_ds.reshape(-1, self.latent_dim*self.latent_dim)
                ).reshape(batch_size, self.latent_dim, self.latent_dim)
        
    def _make_all_L_and_S(self, model_type):
        """Returns L_d and S_d for all domains"""
        d = torch.arange(self.n_domains).to(self.device)
        # Calculate L_d
        L_matrix = torch.zeros(self.n_domains-1, self.latent_dim, self.latent_dim).to(self.device)
        # Fill L_values for d != 0 (Note, for d=0 the L_matrix is all zeros)
        L_d_nonzero_values = self.d_to_L_and_S_nonshared_embeddings_dict[model_type]['L'](d[1:])
        L_matrix[:, self.L_nonshared_idxs[:, 0], self.L_nonshared_idxs[:, 1]] = L_d_nonzero_values
        # add the matrix of zeros for d=0
        L_matrix = torch.cat([torch.zeros(1, self.latent_dim, self.latent_dim).to(self.device), L_matrix], dim=0)
        # Fill S_values for d != 0
        S_matrix = torch.eye(self.latent_dim).unsqueeze(0).tile((self.n_domains-1, 1, 1)).to(self.device)
        S_matrix[:, self.S_nonshared_idxs[:, 0], self.S_nonshared_idxs[:, 1]] =  \
                self.d_to_L_and_S_nonshared_embeddings_dict[model_type]['S'](d[1:])
        # add the identity matrix for d=0
        S_matrix = torch.cat([torch.eye(self.latent_dim).unsqueeze(0).to(self.device), S_matrix], dim=0)
        return L_matrix, S_matrix
    

# class F_VAE_soft_can(F_VAE_can):
#     # A subclass of the F_VAE_auto_spa_can model which is similar except F_{d=0} is not restricted to be identity
#     def __init__(self, config):
#         super().__init__(config)
#
#     def _make_all_L_and_S(self, model_type):
#         """Returns L_d and S_d for all domains"""
#         d = torch.arange(self.n_domains).to(self.device)
#         # Calculate L_d
#         L_matrix = torch.zeros(self.n_domains, self.latent_dim, self.latent_dim).to(self.device)
#         # Fill L_d lower trapezoidal values for each domain
#         L_matrix[:, self.L_nonshared_idxs[:, 0], self.L_nonshared_idxs[:, 1]] = \
#             self.d_to_L_and_S_nonshared_embeddings_dict[model_type]['L'](d)
#         # Fill the last k_spa diagonal entries in S_d for each domain
#         S_matrix = torch.eye(self.latent_dim).unsqueeze(0).tile((self.n_domains, 1, 1)).to(self.device)
#         S_matrix[:, self.S_nonshared_idxs[:, 0], self.S_nonshared_idxs[:, 1]] =  \
#                 self.d_to_L_and_S_nonshared_embeddings_dict[model_type]['S'](d)
#         return L_matrix, S_matrix
#

class F_VAE_relax_can(F_VAE_can):
    """A sparse ILD which has interventions on the last `k_spa` nodes of the latent space"""
    def __init__(self, config):
        super().__init__(config)

    def _setup_models(self, config):
        """Setting up the models for the F_d_mu, F_d_sigma, and F_d_decode 
        (where each F_d_* has a L_d lower lower triangular matrix, S_d diagonal matrix, and bias)"""
        super()._setup_models(config)
        # note, there is not shared parameters can model, so we have to make it here
        self.L_shared_idxs = torch.tensor(
            sum([[(i,j) for j in range(i)] for i in range(self.latent_dim-config.k_spa)], [])
            )
        self.S_shared_idxs = torch.tensor([[i, i] for i in range(self.latent_dim-config.k_spa)])
        # Unfortunately, you can't have nested parameter dicts, so we have to have a seperate dict for L and S
        self.L_shared_dict = nn.ParameterDict({
            model_type: torch.nn.parameter.Parameter(torch.randn(self.L_shared_idxs.shape[0]))
            for model_type in ['mu', 'sigma', 'decode']
        })
        self.S_shared_dict = nn.ParameterDict({
            model_type: torch.nn.parameter.Parameter(torch.randn(self.S_shared_idxs.shape[0]))
            for model_type in ['mu', 'sigma', 'decode']
        })
    
    def _make_all_L_and_S(self, model_type):
        """Returns L_d and S_d for all domains"""
        d = torch.arange(self.n_domains).to(self.device)

        # Calculate L_d
        L_matrix = torch.zeros(self.n_domains, self.latent_dim, self.latent_dim).to(self.device)
        # Fill the M-k lower lower triangular values which are shared across domains
        L_matrix[:, self.L_shared_idxs[:, 0], self.L_shared_idxs[:, 1]] = \
            self.L_shared_dict[model_type].repeat(self.n_domains, 1)
        # Fill L_d lower trapezoidal values for each domain
        L_matrix[:, self.L_nonshared_idxs[:, 0], self.L_nonshared_idxs[:, 1]] = \
            self.d_to_L_and_S_nonshared_embeddings_dict[model_type]['L'](d)
        
        # Calculate S_d
        S_matrix = torch.eye(self.latent_dim).unsqueeze(0).tile((self.n_domains, 1, 1)).to(self.device)
        # Fill the first M-k_spa diagonal entries in S_d which are shared across domain
        S_matrix[:, self.S_shared_idxs[:, 0], self.S_shared_idxs[:, 1]] =  \
                self.S_shared_dict[model_type].repeat(self.n_domains, 1)
        # Fill the last k_spa diagonal entries in S_d for each domain
        S_matrix[:, self.S_nonshared_idxs[:, 0], self.S_nonshared_idxs[:, 1]] =  \
                self.d_to_L_and_S_nonshared_embeddings_dict[model_type]['S'](d)
        return L_matrix, S_matrix

class F_VAE_dense(F_VAE_can):
    # A subclass of the F_VAE_auto_spa_can model which has full dependence on domain (i.e. no shared terms across d)
    def __init__(self, config):
        """f(e|d) = A(d) @ epsilon + b(d) = z"""
        super().__init__(config)

    def _setup_models(self, config):
        """Setting up the models for the L_d lower lower triangular matrix, S_d diagonal matrix, and bias"""
        # getting the indices of the lower lower triangular matrix starting at row N-k
        self.L_nonshared_idxs = torch.tensor(sum([[(i,j) for j in range(i)] 
                                                    for i in range(0, self.latent_dim)], []))
        # getting the indices of the diagonal matrix starting at row N-k
        self.S_nonshared_idxs = torch.tensor([[i, i] for i in range(0, self.latent_dim)])
        # setting up domain specific bias term
        self.shared_bias_terms = torch.randn(0).to(self.device)
        self.d_to_domain_specific_bias_lookup = nn.Embedding(self.n_domains, config.latent_dim)

        # Building the F_d_mu, F_d_sigma, and F_d_decode embeddings
        self.d_to_L_and_S_nonshared_embeddings_dict = nn.ModuleDict({
            model_type: nn.ModuleDict(
                {'L': nn.Embedding(self.n_domains, self.L_nonshared_idxs.shape[0]),
                 'S': nn.Embedding(self.n_domains, self.S_nonshared_idxs.shape[0])}
                )
                    for model_type in ['mu', 'sigma', 'decode']
        })
        # setting up bias terms (these have to be in two separate dicts since one is a parameter and one is a module)
        self.d_to_nonshared_bias_dict = nn.ModuleDict({  # not shared across domains
            model_type: nn.Embedding(self.n_domains, config.latent_dim) 
                for model_type in ['mu', 'sigma', 'decode']
        })
        # there are no shared bias terms for full, but this still needs to be setup to work with the spa code
        self.d_to_shared_bias_dict = nn.ParameterDict({
            model_type: torch.nn.parameter.Parameter(torch.randn(0).to(self.device))  # an empty tensor
                for model_type in ['mu', 'sigma', 'decode']
        })
        return None
        
    def _make_all_L_and_S(self, model_type):
        d = torch.arange(self.n_domains).to(self.device)
        batch_size = d.shape[0]
        # Calculate L_d
        L_matrix = torch.zeros(batch_size, self.latent_dim, self.latent_dim).to(self.device)
        # Fill L_values for each d
        L_d_nonzero_values = self.d_to_L_and_S_nonshared_embeddings_dict[model_type]['L'](d)
        L_matrix[:, self.L_nonshared_idxs[:, 0], self.L_nonshared_idxs[:, 1]] = L_d_nonzero_values
        # Fill S_values for each d
        S_matrix = torch.eye(self.latent_dim).unsqueeze(0).tile((batch_size, 1, 1)).to(self.device)
        S_d_nonzero_values = self.d_to_L_and_S_nonshared_embeddings_dict[model_type]['S'](d)
        S_matrix[:, self.S_nonshared_idxs[:, 0], self.S_nonshared_idxs[:, 1]] = S_d_nonzero_values
        return L_matrix, S_matrix
