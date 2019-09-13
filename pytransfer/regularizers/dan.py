# # -*- coding: utf-8 -*-
from torch import nn
from pytransfer.regularizers import _DiscriminatorBasedReg


class DANReguralizer(_DiscriminatorBasedReg):
    """ Domain advesarial reguralization for learning invariant representations.

    This is a special case of MultipleDAN (num_discriminator=1 and KL_weight=0).

    Reference
    ---------
    https://arxiv.org/abs/1505.07818
    http://papers.nips.cc/paper/6661-controllable-invariance-through-adversarial-feature-learning

    """

    def forward(self, X, y, d):
        d_pred = self.predict(X)
        d_loss = self.criterion(d_pred, d)
        return -1 * d_loss

    def predict(self, X):
        z = self.learner.E(X)
        pred = self.D(z)
        return nn.LogSoftmax(dim=-1)(pred)

    def loss(self, X, y, d):
        d_pred = self.predict(X)
        d_loss = self.criterion(d_pred, d)
        return -1 * d_loss

    def _D_loss(self, X, y, d):
        return -1 * self(X, y, d)
