import torch
import torch.nn as nn
import torch.nn.functional as functional

import numpy as np



class Algorithm(nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    def __init__(self, config):
        super(Algorithm, self).__init__()


    def update(self, x, y):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, config):
        super(ERM, self).__init__(config)

        input_dim = config.latent_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )
        self.model = self.model.to(config.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.erm_lr,
        )

    def update(self, x, y):
        loss = functional.cross_entropy(self.predict(x), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict(self, x):
        return self.model(x)

class ERM_ILD(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, config, ild):
        super(ERM_ILD, self).__init__(config)

        input_dim = config.latent_dim - config.k_spa
        self.k_spa = config.k_spa
        self.ild = ild

        self.model = nn.Sequential(
            nn.Linear(input_dim, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )
        self.model = self.model.to(config.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.erm_lr,
        )

    def update(self, x, y):

        loss = functional.cross_entropy(self.predict(x), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict(self, x):
        with torch.no_grad():
            x = self.ild.G.inverse(x)[:,:-self.k_spa]
        return self.model(x)