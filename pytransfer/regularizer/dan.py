# # -*- coding: utf-8 -*-
from collections import OrderedDict
import numpy as np
from sklearn import metrics

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

    def update(self, on_gpu=False):
        if self.stop_update:
            return None

        for _ in range(self.K):
            self.optimizer.zero_grad()
            X, _, d = self.get_batch(on_gpu)
            d_loss = self.d_loss(self.feature_extractor(X), _, d)
            d_loss.backward()
            self.optimizer.step()

    def parameters(self):
        return self.D.parameters()

    def _evaluate(self, loader, nb_batch, da_flag=False, device='cpu'):
        if nb_batch is None:
            nb_batch = len(loader)
        self.eval()
        targets = []
        preds = []
        loss = 0
        for i, (X, y, d) in enumerate(loader):
            with torch.no_grad():
                X = X.float()
                target = d.long()
                if not da_flag:
                    if len(np.unique(target.data.cpu())) <= 1:
                        continue
                z = self.feature_extractor(X)
                pred = self(z)
                loss += self.loss(z, y, target).item()
                pred = np.argmax(pred.data.cpu(), axis=1)
                targets.append(d.numpy())
                preds.append(pred.numpy())
                if i+1 == nb_batch:
                    break
        loss /= nb_batch

        result = OrderedDict()
        if len(targets) == 0:
            result['accuracy'] = np.nan
            result['f1macro'] = np.nan
            result['loss'] = np.nan
            return result
        target = np.concatenate(targets)
        pred = np.concatenate(preds)
        result['accuracy'] = metrics.accuracy_score(target, pred)
        result['f1macro'] = metrics.f1_score(target, pred, average='macro')
        result['loss'] = loss
        self.train()
        return result

    def validation_step(self, batch, batch_idx):
        X, y, d = batch
        z = self.feature_extractor(X)
        pred = self(z)
        loss = self.loss(z, y, d).item()
        d_hat = torch.argmax(pred, dim=1)
        acc = torch.sum(d == d_hat).item() / (len(d) * 1.0)

        return {'loss': loss, 'acc': acc}
