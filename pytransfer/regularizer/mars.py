# # -*- coding: utf-8 -*-

import torch
from torch import nn


from . import _Reguralizer
from .utils import Discriminator


class EnsembleDAN(_Reguralizer):
    def __init__(self, feature_extractor, num_discriminator, discriminator_config=None, K=1, KL_weight=0.0):
        """ Initialize dan base reguralizer.

        Parameter
        ---------
        num_discriminator : int
          # discrimnator for ensembling
        discriminator_config : dict
          configuration file for the discriminator
        K : int
          the # update of D in each iterations
        KL_weight : float
          the weight parameter for KL divergence
        """

        super(EnsembleDAN, self).__init__()

        discriminator_config['use_softmax'] = False
        self.D = [Discriminator(**discriminator_config).cuda() for i in range(num_discriminator)]
        self.num_output = self.D[0].num_domains

        self.feature_extractor = feature_extractor
        self.K = K
        self.criterion = nn.NLLLoss()
        self.loader = None
        self.KL_weight = 1.0

    def forward(self, z, d):
        return -1 * (nn.NLLLoss()(self.mean_prob(z), d))

    def mean_prob(self, z):
        g = torch.stack([_D(z) for _D in self.D], dim=-1)
        return nn.functional.log_softmax(g.mean(dim=-1), dim=1)

    def update(self, on_gpu=False):
        # All the discriminators are trained using same batch of data.
        for _ in range(self.K):
            self.optimizer.zero_grad()
            X, _, d = self.get_batch(on_gpu)
            z = self.feature_extractor(X).data
            loss = 0
            mean_prob = self.mean_prob(z).data
            for _D in self.D:
                d_pred = nn.functional.log_softmax(_D(z), dim=1)
                loss += nn.NLLLoss()(d_pred, d)
                kl_loss = nn.KLDivLoss(reduction="sum")(d_pred, mean_prob)
                loss -= self.KL_weight * kl_loss
            loss.backward()
            self.optimizer.step()

    def parameters(self):
        weights = []
        for _D in self.D:
            weights += _D.parameters()

        return weights

    def validation_step(self, batch, batch_idx):
        X, y, d = batch
        z = self.feature_extractor(X)
        mean_prob = self.mean_prob(z)
        loss = self(z, d)
        d_hat = torch.argmax(mean_prob, dim=1)
        acc = torch.sum(d == d_hat).item() / (len(d) * 1.0)
        d_mean = torch.exp(mean_prob).mean(dim=0)
        d_ent_reg = torch.sum(-d_mean*torch.log(d_mean))
        result = {'loss': loss.item(), 'acc': acc, 'entropy': d_ent_reg.item()}
        for i, _D in enumerate(self.D):
            d_pred = nn.functional.log_softmax(_D(z), dim=1)
            loss = nn.NLLLoss()(d_pred, d)
            d_hat = torch.argmax(d_pred, dim=1)
            acc = torch.sum(d == d_hat).item() / (len(d) * 1.0)
            kl_loss = nn.KLDivLoss(reduction="sum")(d_pred, mean_prob)
            d_mean = torch.exp(d_pred).mean(dim=0)
            d_ent_reg = torch.sum(-d_mean*torch.log(d_mean))
            _result = {
                'loss-{}'.format(i): loss.item(),
                'acc-{}'.format(i): acc,
                'kl-{}'.format(i): kl_loss.item(),
                'entropy-{}'.format(i): d_ent_reg.item()
            }
            result.update(_result)
        return result
