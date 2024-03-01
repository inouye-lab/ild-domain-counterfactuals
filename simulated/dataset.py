"""Pytorch Dataset object that loads MNIST and SVHN. It returns x,y,s where s=0 when x,y is taken from MNIST."""
import os
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms

class SimulatedDataset(data_utils.Dataset):
    def __init__(self,
                 num_samples=5000,
                 n_domains=3,
                 latent_dim=4,
                 int_set=[2, 3],
                 model_seed=0,
                 noise_seed=1,
                 relu_slope=0.5,
                 device=None,
                 normalization_latent_dict=None,
                 bias_scale=1.0):
        self.num_samples = num_samples
        self.n_domains = n_domains
        self.domain_list = list(range(self.n_domains))
        self.leaky_relu = torch.nn.LeakyReLU(relu_slope)

        self.noise_seed = noise_seed
        self.model_seed = model_seed
        self.device = device

        self.latent_dim = latent_dim
        self.int_set = int_set
        self.normalization_latent_dict = normalization_latent_dict
        self.bias_scale = bias_scale

        self.noise_rng = np.random.RandomState(self.noise_seed)
        self.G = self._generate_g()
        self.A_dict = {}

        self.observed, self.latent, self.domain, self.eps, self.label = self._get_data()

        if device is not None:
            # G needs to be on cuda for the gen_gt_cf function, but this might cause problems with parallelization
            # TODO: check how to get around this...
            self.G = self.G.to(device)

    def noise_to_z(self, eps, d, normalize=True):
        latent = self.F(eps, d)
        if normalize:
            self.normalization_latent_dict['scale'] = self.normalization_latent_dict['scale'].to(latent.device)
            latent = latent / self.normalization_latent_dict['scale']
        return latent

    def z_to_x(self, z):
        x = torch.matmul(z, self.G)  # x = z @ G
        return x

    def gen_gt_cf(self, eps, d_prime):
        assert d_prime in self.domain_list, 'd_prime must be in domain_list, got {d_prime}'
        z_d_prime = self.noise_to_z(eps, d_prime)
        x_d_prime = self.z_to_x(z_d_prime)
        return x_d_prime

    def gen_gt_cf_z(self, eps, d_prime):
        assert d_prime in self.domain_list, 'd_prime must be in domain_list, got {d_prime}'
        z_d_prime = self.noise_to_z(eps, d_prime)
        return z_d_prime

    def __len__(self):
        return len(self.observed)

    def __getitem__(self, index):
        x = self.observed[index]
        d = self.domain[index]
        eps = self.eps[index]

        return x, d, eps

    def _normalize_matrx(self, A):
        dim = A.shape[0]
        det = torch.det(A)
        if det >= 0:
            A = A / (1 * det) ** (1 / dim)
        else:
            A = -A / (-1 * det) ** (1 / dim)
        return A

    def _get_data(self):
        latent = []
        domain = []
        eps = []
        label = []
        # generate latent data for each domain
        for d in range(self.n_domains):
            latent_temp, eps_temp = self._build_latent_domain_dataset(d, normalize=False)  # we will normalize later
            latent.append(latent_temp)
            eps.append(eps_temp)
            domain_temp = torch.ones(self.num_samples) * d
            domain.append(domain_temp)
            label_temp = (latent_temp[:, 1] < 0).long()
            label.append(label_temp)

        latent = torch.cat(latent)
        eps = torch.cat(eps)
        domain = torch.cat(domain).to(torch.int64)
        label = torch.cat(label)

        if not self.normalization_latent_dict:
            self.normalization_latent_dict = {}
            self.normalization_latent_dict['scale'] = latent.std(dim=0)
            self.normalization_latent_dict['scale'] = self.normalization_latent_dict['scale'].to(eps.device)
        else:
            pass
        self.normalization_latent_dict['scale'] = self.normalization_latent_dict['scale'].to(latent.device)
        latent = latent  / self.normalization_latent_dict['scale']

        # generate observed data
        observed = self.z_to_x(latent)
        return observed, latent, domain, eps, label

    def _build_latent_domain_dataset(self, d, normalize=False):
        eps = torch.Tensor(self.noise_rng.randn(self.num_samples, self.latent_dim))
        z = self.noise_to_z(eps, d, normalize)
        return z, eps

    def _generate_g(self):
        """build a pos-def G matrix \in R^{n_dim,n_dim} which has all singular_values = 1"""
        static_rng = np.random.RandomState(self.model_seed)
        G = torch.Tensor(
            static_rng.randn(self.latent_dim,
                             self.latent_dim))  # makes sure we will have the same A matrix for each call
        # G, _ = torch.linalg.qr(G)
        G = self._normalize_matrx(G)
        return G

    def F(self, eps, d):
        """
        Z = ZA + eps
        B = I - A
        Z = eps B^{-1}
        """
        device = eps.device

        global_rng = np.random.RandomState(self.model_seed)
        global_A = torch.Tensor(global_rng.randn(self.latent_dim, self.latent_dim))
        global_A = torch.triu(global_A)
        global_A = global_A - torch.diag(torch.diag(global_A))

        rng = np.random.RandomState(self.model_seed *1000 + d)
        tA = torch.Tensor(rng.randn(self.latent_dim, self.latent_dim))
        tA = torch.triu(tA)
        tA = tA - torch.diag(torch.diag(tA))
        A = global_A.clone()
        A[:, self.int_set] = tA[:, self.int_set]

        self.A_dict[d] = A.clone()

        A = A.to(device)
        F = torch.inverse(torch.eye(self.latent_dim).to(device) - A)
        z = torch.matmul(eps, F)
        # add bias
        bias = torch.Tensor(( 4 *rng.rand(1).astype('float32' ) -2 ) *np.sqrt(self.latent_dim /len(self.int_set))).to \
            (eps.device)
        z[:, self.int_set] += bias * self.bias_scale
        # add leaky
        z = self.leaky_relu(z)
        return z