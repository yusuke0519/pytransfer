# # -*- coding: utf-8 -*-
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

    def __init__(self, feature_extractor, D=None, discriminator_config=None, K=1):
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
        print(self.D)
        self.num_output = self.D.num_domains
        # TODO: DANReguralizer should not assume that D has an attribute num_domain

        self.K = K
        self.criterion = nn.NLLLoss()
        self.loader = None
        self.feature_extractor = feature_extractor

    def forward(self, z):
        return self.D(z)

    def loss(self, z, _, d):
        """ Loss for the encoder update
        """
        d_pred = self(z)
        d_loss = self.criterion(d_pred, d)
        return -1 * d_loss

    def d_loss(self, z, _, d):
        """ Loss for the discriminator update
        """
        d_pred = self(z)
        d_loss = self.criterion(d_pred, d)
        return d_loss

    def update(self):
        if self.stop_update:
            return None


        for _ in range(self.K):
            self.optimizer.zero_grad()
            X, _, d = self.get_batch()
            d_loss = self.d_loss(self.feature_extractor(X), _, d)
            d_loss.backward()
            self.optimizer.step()

    def parameters(self):
        return self.D.parameters()
