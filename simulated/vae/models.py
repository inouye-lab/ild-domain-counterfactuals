import torch
import torch.nn as nn
import torch.nn.functional as functional
import normflows as nf
import numpy as np

class VAEAuto(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.f_type == 'dense':
            self.F = F_VAE_auto_full(config)
            self.G = Gmap(config.obs_dim, config.latent_dim, config.leaky_relu_slope)
        elif config.f_type == 'cano':
            self.F = F_VAE_auto_spa_can(config)
            self.G = Gmap(config.obs_dim, config.latent_dim, config.leaky_relu_slope)
        elif config.f_type == 'relax_cano':
            self.F = F_VAE_auto_spa(config)
            self.G = Gmap(config.obs_dim, config.latent_dim, config.leaky_relu_slope)
        else:
            raise ValueError('f_type not implemented')

    def forward(self, eps, d):
        z = self.F.eps_to_z(eps, d)
        x = self.G.z_to_x(z, d)
        return x

    def inverse(self, x, d):
        z = self.G.x_to_z(x, d)
        eps = self.F.z_to_eps(z, d)
        return eps

class Gmap(nn.Module):
    """If no_leaky_relu is True, then G is a linear transformation such that x = G_mat @ z
    If no_leaky_relu is False, then G takes the form of: x = G_mat @ LeakyReLU(z)."""
    def __init__(self, obs_dim, latent_dim, leaky_relu_slope):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, obs_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(obs_dim, latent_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(latent_dim, latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(latent_dim, obs_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(obs_dim, obs_dim))

    def x_to_z(self, x, d=None):
        return self.encoder(x)
    def z_to_x(self, z, d=None):
        return self.decoder(z)

class GFlowmap(nn.Module):
    """If no_leaky_relu is True, then G is a linear transformation such that x = G_mat @ z
    If no_leaky_relu is False, then G takes the form of: x = G_mat @ LeakyReLU(z)."""
    def __init__(self, latent_dim, depth=4):
        super().__init__()
        self.depth = depth

        # this is not really useful
        base = nf.distributions.base.DiagGaussian(2)
        # Define list of flows
        flows = []
        for i in range(depth):
            # Neural network with two hidden layers having 64 units each
            # Last layer is initialized by zeros making training more stable
            param_map = nf.nets.MLP([latent_dim//2, 64, 64, latent_dim], init_zeros=True)
            # Add flow layer
            flows.append(nf.flows.AffineCouplingBlock(param_map))
            # Swap dimensions
            flows.append(nf.flows.Permute(latent_dim, mode='swap'))
        # Construct flow model
        model = nf.NormalizingFlow(base, flows)
        self.model = model
    def x_to_z(self, x, d=None):
        return self.model.inverse(x)

    def z_to_x(self, z, d=None):
        return self.model(z)

class F_VAE_auto_spa_can(nn.Module):
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
            sum([[(i, j) for j in range(i)] for i in range(self.latent_dim - config.k_spa, self.latent_dim)], [])
        )
        # getting the indices of the diagonal matrix starting at row N-k
        self.S_nonshared_idxs = torch.tensor([[i, i] for i in range(self.latent_dim - config.k_spa, self.latent_dim)])
        self.S_shared_idxs = torch.tensor([[i, i] for i in range(self.latent_dim - config.k_spa)])
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
            model_type: torch.nn.parameter.Parameter(torch.randn(self.latent_dim - config.k_spa))
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
            d, F_ds.reshape(-1, self.latent_dim * self.latent_dim)
        ).reshape(batch_size, self.latent_dim, self.latent_dim)

    def _make_all_L_and_S(self, model_type):
        """Returns L_d and S_d for all domains"""
        d = torch.arange(self.n_domains).to(self.device)
        # Calculate L_d
        L_matrix = torch.zeros(self.n_domains - 1, self.latent_dim, self.latent_dim).to(self.device)
        # Fill L_values for d != 0 (Note, for d=0 the L_matrix is all zeros)
        L_d_nonzero_values = self.d_to_L_and_S_nonshared_embeddings_dict[model_type]['L'](d[1:])
        L_matrix[:, self.L_nonshared_idxs[:, 0], self.L_nonshared_idxs[:, 1]] = L_d_nonzero_values
        # add the matrix of zeros for d=0
        L_matrix = torch.cat([torch.zeros(1, self.latent_dim, self.latent_dim).to(self.device), L_matrix], dim=0)
        # Fill S_values for d != 0
        S_matrix = torch.eye(self.latent_dim).unsqueeze(0).tile((self.n_domains - 1, 1, 1)).to(self.device)
        S_matrix[:, self.S_nonshared_idxs[:, 0], self.S_nonshared_idxs[:, 1]] = \
            self.d_to_L_and_S_nonshared_embeddings_dict[model_type]['S'](d[1:])
        # add the identity matrix for d=0
        S_matrix = torch.cat([torch.eye(self.latent_dim).unsqueeze(0).to(self.device), S_matrix], dim=0)
        return L_matrix, S_matrix

class F_VAE_auto_spa(F_VAE_auto_spa_can):
    """A sparse ILD which has interventions on the last `k_spa` nodes of the latent space"""

    def __init__(self, config):
        super().__init__(config)

    def _setup_models(self, config):
        """Setting up the models for the F_d_mu, F_d_sigma, and F_d_decode
        (where each F_d_* has a L_d lower lower triangular matrix, S_d diagonal matrix, and bias)"""
        super()._setup_models(config)
        # note, there is not shared parameters can model, so we have to make it here
        self.L_shared_idxs = torch.tensor(
            sum([[(i, j) for j in range(i)] for i in range(self.latent_dim - config.k_spa)], [])
        )
        self.S_shared_idxs = torch.tensor([[i, i] for i in range(self.latent_dim - config.k_spa)])
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
        S_matrix[:, self.S_shared_idxs[:, 0], self.S_shared_idxs[:, 1]] = \
            self.S_shared_dict[model_type].repeat(self.n_domains, 1)
        # Fill the last k_spa diagonal entries in S_d for each domain
        S_matrix[:, self.S_nonshared_idxs[:, 0], self.S_nonshared_idxs[:, 1]] = \
            self.d_to_L_and_S_nonshared_embeddings_dict[model_type]['S'](d)
        return L_matrix, S_matrix


class F_VAE_auto_full(F_VAE_auto_spa_can):
    # A subclass of the F_VAE_auto_spa_can model which has full dependence on domain (i.e. no shared terms across d)
    def __init__(self, config):
        """f(e|d) = A(d) @ epsilon + b(d) = z"""
        super().__init__(config)

    def _setup_models(self, config):
        """Setting up the models for the L_d lower lower triangular matrix, S_d diagonal matrix, and bias"""
        # getting the indices of the lower lower triangular matrix starting at row N-k
        self.L_nonshared_idxs = torch.tensor(sum([[(i, j) for j in range(i)]
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
