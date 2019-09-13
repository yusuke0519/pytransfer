# # -*- coding: utf-8 -*-
from collections import OrderedDict

import numpy as np
from sklearn import metrics
from torch import nn
from torch.utils import data
from torch.autograd import Variable
from pytransfer.regularizers.utils import Discriminator


class _Reguralizer(nn.Module):
    def set_loader(self, dataset, batch_size):
        self.loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return self.loader

    def get_batch(self, as_variable=True):
        """ Get a batch of data

        TODO: this function should be removed and the data should be specified outside of the class
        """
        assert self.loader is not None, "Please set loader before call this function"
        X, y, d = self.loader.__iter__().__next__()
        if as_variable:
            X = Variable(X.float().cuda())
            y = Variable(y.long().cuda())
            d = Variable(d.long().cuda())
        return X, y, d

    def _evaluate(self, loader, nb_batch):
        if nb_batch is None:
            nb_batch = len(loader)
        self.eval()
        targets = []
        preds = []
        loss = 0
        for i, (X, y, d) in enumerate(loader):
            X = Variable(X.float().cuda(), volatile=True)
            target = Variable(d.long().cuda(), volatile=True)
            if len(np.unique(target.data.cpu())) <= 1:
                continue
            pred = self.predict(X)
            loss += self(X, y, target).data[0]
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

    def forward(self, X, y, d):
        """ Returan reguralization loss.

        Parameters
        ----------
        X : torch.FloatTensor
        y : torch.LongTensor
        d : torch.LongTensor

        Returns
        -------
        loss : loss of the regularizer. The loss is minimized by trainer.

        """
        raise NotImplementedError()

    def predict(self, X):
        """ Predict with internal neural networks

        Parameters
        ----------
        X : torch.FloatTensor
        y : torch.LongTensor
        d : torch.LongTensor
        """
        raise NotImplementedError()


class _DiscriminatorBasedReg(_Reguralizer):
    def __init__(self, learner, D=None, discriminator_config=None, K=1):
        super(_DiscriminatorBasedReg, self).__init__()

        self.stop_update = D is not None  # if D is shared with others, then not update here
        if D is None:
            D = Discriminator(**discriminator_config)
        self.D = D.cuda()
        self.num_output = self.D.num_domains

        self.learner = learner
        self.K = K
        self.criterion = nn.NLLLoss()
        self.loader = None

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def _D_loss(self, X, y, d):
        raise NotImplementedError()

    def parameters(self):
        return self.D.parameters()

    def update(self):
        if self.stop_update:
            return None

        for _ in range(self.K):
            self.optimizer.zero_grad()
            X, _, d = self.get_batch()
            d_loss = self._D_loss(X, _, d)
            d_loss.backward()
            self.optimizer.step()
