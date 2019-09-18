# # -*- coding: utf-8 -*-
from torch import nn

from pytransfer.reguralizer.utils import Discriminator
from pytransfer.reguralizer import _Reguralizer


class DANReguralizer(_Reguralizer):
    """ Domain advesarial reguralization for learning invariant representations.

    This is a special case of MultipleDAN (num_discriminator=1 and KL_weight=0).

    Reference
    ---------
    https://arxiv.org/abs/1505.07818
    http://papers.nips.cc/paper/6661-controllable-invariance-through-adversarial-feature-learning

    """

    def __init__(self, learner, D=None, discriminator_config=None, K=1):
        """
        Initialize dan base reguralizer

        Parameter
        ---------
        learner : instance of Learner
          TBA
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
        self.D = D.cuda()
        print(self.D)
        self.num_output = self.D.num_domains
        # TODO: DANReguralizer should not assume that D has an attribute num_domain

        self.learner = learner
        self.K = K
        self.criterion = nn.NLLLoss()
        self.loader = None

    def forward(self, X):
        z = self.learner.E(X)
        return self.D(z)

    def loss(self, X, y, d):
        d_pred = self(X)
        d_loss = self.criterion(d_pred, d)
        return -1 * d_loss

    def d_loss(self, X, y, d):
        z = self.learner.E(X)
        d_pred = self.D(z)
        d_loss = self.criterion(d_pred, d)
        return d_loss

    def update(self):
        if self.stop_update:
            return None

        for _ in range(self.K):
            self.optimizer.zero_grad()
            X, _, d = self.get_batch()
            d_loss = self.d_loss(X, _, d)
            d_loss.backward()
            self.optimizer.step()

    def parameters(self):
        return self.D.parameters()
