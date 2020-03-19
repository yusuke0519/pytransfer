# # -*- coding: utf-8 -*-
# from collections import OrderedDict
# import numpy as np
# from sklearn import metrics

import torch
from torch import nn

from pytransfer.regularizer.utils import Discriminator
from pytransfer.regularizer import _Reguralizer


class DANReguralizer(_Reguralizer):
    """ Domain advesarial reguralization for learning invariant representations.

    This is a special case of MultipleDAN (num_discriminator=1 and KL_weight=0).

    Reference
    ---------
    https://arxiv.org/abs/1505.07818
    http://papers.nips.cc/paper/6661-controllable-invariance-through-adversarial-feature-learning

    """

    def __init__(self, feature_extractor, D=None, discriminator_config=None, K=1, max_ent=False):
        """
        Initialize dan base regularizer

        Parameter
        ---------
        discriminator_config : dict
          configuration file for the discriminator
        K : int
          the # update of D in each iterations
        """
        super(DANReguralizer, self).__init__()

        # if D is shared with others, then not update here
        self.stop_update = D is not None
        if D is None:
            D = Discriminator(**discriminator_config)
        self.D = D
        self.num_output = self.D.num_domains
        # TODO: DANReguralizer should not assume that D has an attribute num_domain

        self.K = K
        self.criterion = nn.NLLLoss()
        self.loader = None
        self.feature_extractor = feature_extractor
        self.max_ent = max_ent

    def forward(self, z, d):
        d_pred = self.D(z)
        d_loss = self.criterion(d_pred, d)
        if self.max_ent:
            d_mean = torch.exp(d_pred).mean(dim=0)
            d_ent_reg = torch.sum(-d_mean*torch.log(d_mean))
            return -1 * d_ent_reg
        return -1 * d_loss

    def d_loss(self, z, d):
        """ Loss for the discriminator update
        """
        return -1 * self(z, d)

    def update(self, on_gpu=False):
        if self.stop_update:
            return None

        for _ in range(self.K):
            self.optimizer.zero_grad()
            X, _, d = self.get_batch(on_gpu)
            d_loss = self.d_loss(self.feature_extractor(X), d)
            d_loss.backward()
            self.optimizer.step()

    def parameters(self):
        return self.D.parameters()

    def validation_step(self, batch, batch_idx):
        X, y, d = batch
        z = self.feature_extractor(X)
        pred = self.D(z)
        loss = self(z, d).item()
        d_hat = torch.argmax(pred, dim=1)
        acc = torch.sum(d == d_hat).item() / (len(d) * 1.0)
        d_mean = torch.exp(pred).mean(dim=0)
        d_ent_reg = torch.sum(-d_mean*torch.log(d_mean)).item()

        return {'loss': loss, 'acc': acc, 'entropy': d_ent_reg}
