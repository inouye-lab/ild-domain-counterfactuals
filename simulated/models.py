import torch
import torch.nn as nn
import torch.nn.functional as functional

import numpy as np


def _inverse_batch_matmul(A, x):
    """
    Returns A^{-1} @ x, where A is a square matrix and x is a (possibly batched) vector.
    """
    assert x.ndim in [1, 2], 'x must be a vector or a batch of vectors'
    if x.ndim == 2:
        return torch.linalg.solve(A, x.unsqueeze(-1)).squeeze()
    else:
        return torch.linalg.solve(A, x)


def _build_leaky_A_u_jacobian(A, u, leaky_slope):
    """"Builds the jacobian matrix of the function f = L(A @ x) = z where L is leaky relu and A is NxN matrix.
    This can also be used to build the jacobian of the *inverse* of f, i.e. f^{-1} = A^{-1} @ L^{-1}(z).
    For f, let u_{i} = [Ax]_{i} and for f^{-1}, let u_{i} = [x]_i. Then,
    [J]_{i,j} = dL/du_{i} * du_{i}/d[x]_{j}
    dL/du_{i} = A_{i,j}
    du_{i}/d[x]_{j} = 1 if u_{i} > 0 else leaky_slope
    ==> [J]_{i, j} = A_{i,j} if u_{i} > 0 else leaky_slope * A_{i,j}"""
    u_negative_mask = u < 0  # True if u[i,j] is negative
    if A.ndim == 2 and u.ndim == 2:
        # u is batched but A is not, so repeat A across the batch dimension
        A = A.unsqueeze(0).repeat(u.shape[0], 1, 1)
    J = A.clone()
    J[u_negative_mask] *= leaky_slope
    return J


def _get_log_det(A):
    return torch.log(torch.abs(torch.det(A)))


class G(nn.Module):
    """If no_leaky_relu is True, then G is a linear transformation such that x = G_mat @ z
    If no_leaky_relu is False, then G takes the form of: x = G_mat @ LeakyReLU(z)."""

    def __init__(self, config):
        super().__init__()
        self.no_leaky_relu = config.no_leaky_relu

        self.G = nn.parameter.Parameter(
            torch.eye(config.latent_dim)
        )

        self.bias = nn.parameter.Parameter(torch.zeros(config.latent_dim))

        if not self.no_leaky_relu:
            self.leaky_slope = config.leaky_relu_slope

    def forward(self, z, return_jacobian=False):
        """x = G @ h(z) + b where h is a leaky relu function if no_leaky_relu is False, else h is the identity function."""
        assert z.ndim in [1, 2], 'z must be a vector or a batch of vectors'

        if not self.no_leaky_relu:
            z = functional.leaky_relu(z, self.leaky_slope)

        if z.ndim == 2:
            x = (self.G @ z.unsqueeze(-1)).squeeze(-1) + self.bias
        else:
            x = self.G @ z + self.bias

        if return_jacobian:
            jacobian = _build_leaky_A_u_jacobian(self.G, z, self.leaky_slope) if not self.no_leaky_relu else self.G
            return x, _get_log_det(jacobian)
        else:
            return x

    def inverse(self, x, return_jacobian=False):
        """z = h^{-1}( G^{-1} @ (x) )"""
        assert x.ndim in [1, 2], 'x must be a vector or a batch of vectors'
        G_inv = torch.linalg.inv(self.G)

        x = x - self.bias
        if x.ndim == 2:
            z = (G_inv @ x.unsqueeze(-1)).squeeze(-1)
        else:
            z = G_inv @ x

        if return_jacobian:
            # calculate jacobian of z = h^{-1}( G^{-1} @ (x) ) *before applying h^{-1}*
            # jacobian = _build_leaky_A_u_jacobian(G_inv, z, 1/self.leaky_slope) if not self.no_leaky_relu else self.G
            jacobian = _build_leaky_A_u_jacobian(G_inv, z, 1 / self.leaky_slope) if not self.no_leaky_relu else G_inv

        if not self.no_leaky_relu:
            z = functional.leaky_relu(z, 1 / self.leaky_slope)

        if return_jacobian:
            return z, _get_log_det(jacobian)
        else:
            return z


class F_can(nn.Module):
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

class F_relax_can(F_can):

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


class F_dense(F_can):
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
