import torch
import torch.nn as nn
import torch.nn.functional as functional
import normflows as nf
import numpy as np

def _get_log_det(A):
    return torch.log(torch.abs(torch.det(A)))
def _inverse_batch_matmul(A, x):
    """
    Returns A^{-1} @ x, where A is a square matrix and x is a (possibly batched) vector.
    """
    assert x.ndim in [1, 2], 'x must be a vector or a batch of vectors'
    if x.ndim == 2:
        return torch.linalg.solve(A, x.unsqueeze(-1)).squeeze()
    else:
        return torch.linalg.solve(A, x)
class FlowAuto(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.f_type == 'dense':
            self.F = F_auto_full(config)
            self.G = Gmap(config.latent_dim, config.model_depth)
        elif config.f_type == 'cano':
            self.F = F_auto_spa_can(config)
            self.G = Gmap(config.latent_dim, config.model_depth)
        elif config.f_type == 'relax_cano':
            self.F = F_auto_spa(config)
            self.G = Gmap(config.latent_dim, config.model_depth)
        else:
            raise ValueError('f_type not implemented')

    def forward(self, eps, d):
        z = self.F(eps, d)
        x = self.G.z_to_x(z, d)
        return x

    def inverse(self, x, d):
        z = self.G.x_to_z(x, d)
        eps = self.F.inverse(z, d)
        return eps


class Gmap(nn.Module):
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
    def x_to_z(self, x, d=None,return_jacobian=False):
        if return_jacobian:
            return self.model.inverse_and_log_det(x)
        else:
            return self.model.inverse(x)

    def z_to_x(self, z, d=None):
        return self.model(z)


class F_auto_spa_can(nn.Module):
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
        self.no_model_bias = config.no_model_bias
        self.latent_dim = config.latent_dim
        self.leaky_relu_slope = config.leaky_relu_slope
        self.no_model_s = config.no_model_s
        # setting up projection from dim to S, L, and bias
        self._setup_models(config)

    def _setup_models(self, config):
        """Setting up the models for the L_d lower lower triangular matrix, S_d diagonal matrix, and bias"""
        assert self.latent_dim > 0 and config.k_spa > 0, f'Must have N>=0 and k>=0, got N={self.latent_dim} and k={config.k_spa}'
        assert self.latent_dim > config.k_spa, f'Must have N > k, got N={self.latent_dim} and k={config.k_spa}'
        # getting the indices of the lower lower triangular matrix starting at row N-k
        self.L_nonshared_idxs = torch.tensor(sum([[(i, j) for j in range(i)]
                                                  for i in range(self.latent_dim - config.k_spa, self.latent_dim)], []))
        # getting the indices of the diagonal matrix starting at row N-k
        self.S_nonshared_idxs = torch.tensor([[i, i] for i in range(self.latent_dim - config.k_spa, self.latent_dim)])
        # building projections from domain to L and S

        self.d_to_L_nonshared = nn.Embedding(self.n_domains, self.L_nonshared_idxs.shape[0])
        self.d_to_S_nonshared = nn.Embedding(self.n_domains, self.S_nonshared_idxs.shape[0])

        if not self.no_model_bias:
            # setting up domain specific bias term
            self.shared_bias_terms = torch.nn.parameter.Parameter(torch.randn(self.latent_dim - config.k_spa))
            self.d_to_domain_specific_bias_lookup = nn.Embedding(self.n_domains, config.k_spa)
        return None

    def forward(self, epsilon, d, return_jacobian=False):
        """ f(epsilon, d) = F_d @ epsilon + b_d = z_d"""
        assert epsilon.ndim == 2, f'Epsilon must have shape: batch_dim, latent_dim'
        assert d.ndim == 1, f'd must have shape: batch_dim'
        # Find L_d and S_d, and use these to calculate F_d
        F_matrix = self._make_F(*self._make_L_and_S(d))
        # Calculate z_d

        z = (F_matrix @ epsilon.unsqueeze(-1)).squeeze() + self._make_bias(d)
        if return_jacobian:
            return z, _get_log_det(F_matrix)
        else:
            return z

    def inverse(self, z, d, return_jacobian=False):
        """ f^{-1}(z, d) = F_d^{-1} @ (z_d - b_d) = epsilon"""
        assert z.ndim == 2, f'z must have shape: batch_dim, latent_dim'
        assert d.ndim == 1, f'd must have shape: batch_dim'
        # Find L_d and S_d, and use these to calculate F_d
        F_matrix = self._make_F(*self._make_L_and_S(d))
        # Remove bias from z
        z = z - self._make_bias(d)
        # print(d, self._make_bias(d))
        if return_jacobian:
            # Calculate F_d^{-1} explicitly so we can return it as well
            F_matrix_inv = F_matrix.inverse()
            epsilon = (F_matrix_inv @ z.unsqueeze(-1)).squeeze()
            return epsilon, _get_log_det(F_matrix_inv)
        else:
            # Calculate F_d^{-1} implicitly
            return _inverse_batch_matmul(F_matrix, z)

    def _make_bias(self, d):
        batch_size = d.shape[0]
        if self.no_model_bias:
            bias = torch.zeros(batch_size, self.latent_dim).to(self.device)
        else:
            bias = torch.cat([self.shared_bias_terms.repeat(batch_size, 1),
                              self.d_to_domain_specific_bias_lookup(d)], dim=-1)
        return bias

    def _make_F(self, L_matrix, S_matrix):
        batch_size = L_matrix.shape[0]
        # Calculate F_d
        return torch.linalg.solve(
            torch.eye(self.latent_dim).unsqueeze(0).tile((batch_size, 1, 1)).to(self.device) - L_matrix, S_matrix
        )  # a more numerically stable way of doing (I - L_d)^{-1} @ S_d

    def _make_L_and_S(self, d):
        batch_size = d.shape[0]
        is_d0_mask = d == 0
        non_zero_d_idxs = torch.nonzero(~is_d0_mask)
        # Calculate L_d
        L_matrix = torch.zeros(batch_size, self.latent_dim, self.latent_dim).to(self.device)
        # Fill L_values for d != 0
        L_d_nonzero_values = self.d_to_L_nonshared(d[~is_d0_mask])
        L_matrix[non_zero_d_idxs, self.L_nonshared_idxs[:, 0], self.L_nonshared_idxs[:, 1]] = L_d_nonzero_values
        # Note, for d=0 the L_matrix is all zeros
        # Fill S_values for d != 0
        S_matrix = torch.eye(self.latent_dim).unsqueeze(0).tile((batch_size, 1, 1)).to(self.device)
        if self.no_model_s:
            pass
        else:
            S_d_nonzero_values = self.d_to_S_nonshared(d[~is_d0_mask])
            S_matrix[non_zero_d_idxs, self.S_nonshared_idxs[:, 0], self.S_nonshared_idxs[:, 1]] = S_d_nonzero_values
        return L_matrix, S_matrix


class F_auto_spa(F_auto_spa_can):
    # A subclass of the F_auto_spa_can model which is similar except F_{d=0} is not restricted to be identity
    def __init__(self, config):
        """f(e|d) = A(d) @ epsilon + b(d) = z"""
        super().__init__(config)

    def _make_L_and_S(self, d):
        batch_size = d.shape[0]
        # Calculate L_d
        L_matrix = torch.zeros(batch_size, self.latent_dim, self.latent_dim).to(self.device)
        # Fill L_values for all d
        L_d_nonzero_values = self.d_to_L_nonshared(d)
        L_matrix[:, self.L_nonshared_idxs[:, 0], self.L_nonshared_idxs[:, 1]] = L_d_nonzero_values
        # Fill S_values for all d
        S_matrix = torch.eye(self.latent_dim).unsqueeze(0).tile((batch_size, 1, 1)).to(self.device)
        if self.no_model_s:
            pass
        else:
            S_d_nonzero_values = self.d_to_S_nonshared(d)
            S_matrix[:, self.S_nonshared_idxs[:, 0], self.S_nonshared_idxs[:, 1]] = S_d_nonzero_values
        return L_matrix, S_matrix


class F_auto_true_spa(F_auto_spa_can):

    def __init__(self, config):
        """f(e|d) = A(d) @ epsilon + b(d) = z"""
        super().__init__(config)

    def _setup_models(self, config):
        """Setting up the models for the L_d lower lower triangular matrix, S_d diagonal matrix, and bias"""
        assert self.latent_dim > 0 and config.k_spa > 0, f'Must have N>=0 and k>=0, got N={self.latent_dim} and k={config.k_spa}'
        assert self.latent_dim > config.k_spa, f'Must have N > k, got N={self.latent_dim} and k={config.k_spa}'
        # getting the indices of the L_d lower lower triangular matrix
        self.L_shared_idxs = torch.tensor(
            sum([[(i, j) for j in range(i)] for i in range(self.latent_dim - config.k_spa)], [])
        )
        self.L_nonshared_idxs = torch.tensor(
            sum([[(i, j) for j in range(i)] for i in range(self.latent_dim - config.k_spa, self.latent_dim)], [])
        )
        # getting the indices of the diagonal matrix
        self.S_shared_idxs = torch.tensor([[i, i] for i in range(self.latent_dim - config.k_spa)])
        self.S_nonshared_idxs = torch.tensor([[i, i] for i in range(self.latent_dim - config.k_spa, self.latent_dim)])
        # building projections from domain to L and S
        self.L_shared_lookup = torch.nn.parameter.Parameter(torch.randn(self.L_shared_idxs.shape[0]))
        self.d_to_L_nonshared = nn.Embedding(self.n_domains, self.L_nonshared_idxs.shape[0])
        self.S_shared_lookup = torch.nn.parameter.Parameter(torch.randn(self.S_shared_idxs.shape[0]))
        self.d_to_S_nonshared = nn.Embedding(self.n_domains, self.S_nonshared_idxs.shape[0])
        if not self.no_model_bias:
            # setting up domain specific bias term
            self.shared_bias_terms = torch.nn.parameter.Parameter(torch.randn(self.latent_dim - config.k_spa))
            self.d_to_domain_specific_bias_lookup = nn.Embedding(self.n_domains, config.k_spa)
        return None

    def _make_L_and_S(self, d):
        batch_size = d.shape[0]
        # Calculate L_d
        L_matrix = torch.zeros(batch_size, self.latent_dim, self.latent_dim).to(self.device)
        # Fill L_values for all d
        L_matrix[:, self.L_shared_idxs[:, 0], self.L_shared_idxs[:, 1]] = self.L_shared_lookup.repeat(batch_size, 1)
        L_matrix[:, self.L_nonshared_idxs[:, 0], self.L_nonshared_idxs[:, 1]] = self.d_to_L_nonshared(d)
        # Fill S_values for all d
        S_matrix = torch.eye(self.latent_dim).unsqueeze(0).tile((batch_size, 1, 1)).to(self.device)
        if self.no_model_s:
            pass
        else:
            S_matrix[:, self.S_shared_idxs[:, 0], self.S_shared_idxs[:, 1]] = self.S_shared_lookup.repeat(batch_size, 1)
            S_matrix[:, self.S_nonshared_idxs[:, 0], self.S_nonshared_idxs[:, 1]] = self.d_to_S_nonshared(d)
        return L_matrix, S_matrix


class F_auto_full(F_auto_spa_can):
    # A subclass of the F_auto_spa_can model which has full dependence on domain (i.e. no shared terms across d)
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
        # building projections from domain to L and S

        # Don't use MLP, just use embedding directly
        self.d_to_L_nonshared = nn.Embedding(self.n_domains, self.L_nonshared_idxs.shape[0])
        self.d_to_S_nonshared = nn.Embedding(self.n_domains, self.S_nonshared_idxs.shape[0])

        if not self.no_model_bias:
            # setting up domain specific bias term
            self.shared_bias_terms = torch.randn(0).to(self.device)
            self.d_to_domain_specific_bias_lookup = nn.Embedding(self.n_domains, config.latent_dim)
        return None

    def _make_L_and_S(self, d):
        batch_size = d.shape[0]
        # Calculate L_d
        L_matrix = torch.zeros(batch_size, self.latent_dim, self.latent_dim).to(self.device)
        # Fill L_values for each d
        L_d_nonzero_values = self.d_to_L_nonshared(d)
        L_matrix[:, self.L_nonshared_idxs[:, 0], self.L_nonshared_idxs[:, 1]] = L_d_nonzero_values
        # Fill S_values for each d
        S_matrix = torch.eye(self.latent_dim).unsqueeze(0).tile((batch_size, 1, 1)).to(self.device)
        if self.no_model_s:
            pass
        else:
            S_d_nonzero_values = self.d_to_S_nonshared(d)
            S_matrix[:, self.S_nonshared_idxs[:, 0], self.S_nonshared_idxs[:, 1]] = S_d_nonzero_values
        return L_matrix, S_matrix
