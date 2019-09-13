# # -*- coding: utf-8 -*-
from future.utils import iteritems
import os
from collections import OrderedDict

import numpy as np
from sklearn import metrics
import torch
from torch import nn
from torch.utils import data
from torch.autograd import Variable


class Learner(nn.Module):
    def __init__(self, E, M):
        super(Learner, self).__init__()
        self.E = E
        self.M = M
        self.reguralizers = OrderedDict()
        self.criterion = nn.NLLLoss()

    def forward(self, X):
        z = self.E(X)
        y = self.M(z)
        return y

    def add_reguralizer(self, name, reguralizer, alpha):
        assert name not in self.reguralizers, "name {} is already registered".format(name)
        self.reguralizers[name] = (reguralizer, alpha)

    def update_reguralizers(self):
        for reguralizer, _ in self.reguralizers.values():
            reguralizer.update()

    def loss(self, X, y, d):
        yhat = self(X)
        y_loss = self.criterion(yhat, y)
        loss = y_loss
        for reguralizer, alpha in self.reguralizers.values():
            loss += alpha * reguralizer(X, y, d)
        return loss

    def losses(self, X, y, d):
        yhat = self(X)
        y_loss = self.criterion(yhat, y)
        losses = {}
        losses['y'] = y_loss.data[0]
        for i, (reguralizer, alpha) in enumerate(self.reguralizers.values()):
            losses[i] = reguralizer(X, y, d).data[0]
        return losses

    def set_loader(self, dataset, batch_size):
        self.loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for reguralizer, _ in self.reguralizers.values():
            if reguralizer.loader is None:
                reguralizer.set_loader(dataset, batch_size)
        return self.loader

    def get_batch(self, as_variable=True):
        assert self.loader is not None, "Please set loader before call this function"
        X, y, d = self.loader.__iter__().__next__()
        if as_variable:
            X = Variable(X.float().cuda())
            y = Variable(y.long().cuda())
            d = Variable(d.long().cuda())
        return X, y, d

    def predict_y(self, input_data):
        return self.M(self.E(input_data))

    def evaluate(self, loader, nb_batch=None):
        """
        Evaluate model given data loader

        Parameter
        ---------
        nb_batch : int (default: None)
          # batch to calculate the loss and accuracy
        loader : DataLoader
          data loader

        """
        if nb_batch is None:
            nb_batch = len(loader)
        self.eval()
        result = OrderedDict()

        # evaluate main
        for k, v in iteritems(self._evaluate(loader, nb_batch)):
            result['{}-{}'.format('y', k)] = v

        # evaluate reguralizer
        for name, (reguralizer, _) in iteritems(self.reguralizers):
            for k, v in iteritems(reguralizer._evaluate(loader, nb_batch)):
                result['{}-{}'.format(name, k)] = v

        self.train()
        return result

    def _evaluate(self, loader, nb_batch):
        ys = []
        pred_ys = []
        loss = 0
        criterion = nn.NLLLoss()
        for i, (X, y, d) in enumerate(loader):
            X = Variable(X.float().cuda(), volatile=True)
            target = Variable(y.long().cuda(), volatile=True)
            pred_y = self.predict_y(X)
            loss += criterion(pred_y, target).data[0]
            pred_y = np.argmax(pred_y.data.cpu(), axis=1)
            ys.append(y.numpy())
            pred_ys.append(pred_y.numpy())
            if i+1 == nb_batch:
                break
        loss /= nb_batch
        # print(nb_batch, loss)

        y = np.concatenate(ys)
        pred_y = np.concatenate(pred_ys)

        result = OrderedDict()
        result['accuracy'] = metrics.accuracy_score(y, pred_y)
        result['f1macro'] = metrics.f1_score(y, pred_y, average='macro')
        result['loss'] = loss
        return result

    def save(self, out, prefix=None):
        names = ['E.pth', 'M.pth']
        if prefix is not None:
            names = [prefix + '-' + x for x in names]
        names = [os.path.join(out, x) for x in names]

        for net, name in zip([self.E, self.M], names):
            torch.save(net.state_dict(), name)
        return names
