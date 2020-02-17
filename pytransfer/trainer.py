# # -*- coding: utf-8 -*-
from future.utils import iteritems
import os
from collections import OrderedDict

import numpy as np
from sklearn import metrics
import torch
from torch import nn
from torch.utils import data


class Learner(nn.Module):
    def __init__(self, E, M):
        super(Learner, self).__init__()
        self.E = E
        self.M = M
        self.regularizers = OrderedDict()
        self.criterion = nn.NLLLoss()

    def forward(self, X):
        z = self.E(X)
        y = self.M(z)
        return y

    def add_regularizer(self, name, regularizer, alpha):
        assert name not in self.regularizers, "name {} is already registered".format(name)
        self.regularizers[name] = (regularizer, alpha)

    def update_regularizers(self):
        for regularizer, _ in self.regularizers.values():
            regularizer.update()

    def loss(self, X, y, d):
        yhat = self(X)
        y_loss = self.criterion(yhat, y)
        loss = y_loss
        for regularizer, alpha in self.regularizers.values():
            loss += alpha * regularizer.loss(self.E(X), y, d)
        return loss

    def losses(self, X, y, d):
        yhat = self(X)
        y_loss = self.criterion(yhat, y)
        losses = {}
        losses['y'] = y_loss.item()
        for i, (regularizer, alpha) in enumerate(self.regularizers.values()):
            losses[i] = regularizer.loss(self.E(X), y, d).item()
        return losses

    def set_loader(self, dataset, batch_size):
        self.loader = data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True)
        for regularizer, _ in self.regularizers.values():
            if regularizer.loader is None:
                regularizer.set_loader(dataset, batch_size)
        return self.loader

    def get_batch(self, as_variable=True, device='cpu'):
        assert self.loader is not None, "Please set loader before call this function"
        X, y, d = self.loader.__iter__().__next__()
        if as_variable:
            X = X.float().to(device)
            y = y.long().to(device)
            d = d.long().to(device)
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

        # evaluate regularizer
        for name, (regularizer, _) in iteritems(self.regularizers):
            for k, v in iteritems(regularizer._evaluate(loader, nb_batch)):
                result['{}-{}'.format(name, k)] = v

        self.train()
        return result

    def _evaluate(self, loader, nb_batch):
        ys = []
        pred_ys = []
        loss = 0
        criterion = nn.NLLLoss()
        with torch.no_grad():
            for i, (X, y, d) in enumerate(loader):
                X = X.float()
                target = y.long()
                pred_y = self.predict_y(X)
                loss += criterion(pred_y, target).data[0]
                pred_y = np.argmax(pred_y.data, axis=1)
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


class DALearner(Learner):
    def __init__(self, *args, **kwargs):
        super(DALearner, self).__init__(*args, **kwargs)

    def set_loader(self, dataset, sampler, batch_size):
        self.source_loader = data.DataLoader(
            dataset, batch_size=batch_size, sampler=sampler)  # source only
        self.loader = data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True)  # random sampling
        for regularizer, _ in self.regularizers.values():
            if regularizer.loader is None:
                regularizer.set_loader(dataset, batch_size)
        return self.source_loader, self.loader

    def get_batch(self, as_variable=True):
        assert self.source_loader is not None, "Please set loader before call this function"
        X_s, y_s, _ = self.source_loader.__iter__().__next__()
        assert self.loader is not None, "Please set loader before call this function"
        X, _, d = self.loader.__iter__().__next__()
        if as_variable:
            X = X.float()
            X_s = X_s.float()
            y_s = y_s.long()
            d = d.long()
        return X_s, y_s, X, d

    def loss(self, X_s, y_s, X, d):
        yhat = self(X_s)
        y_loss = self.criterion(yhat, y_s)
        loss = y_loss
        for regularizer, alpha in self.regularizers.values():
            loss += alpha * regularizer.loss(X, y_s, d)
        return loss

    def losses(self, X_s, y_s, X, d):
        yhat = self(X_s)
        y_loss = self.criterion(yhat, y_s)
        losses = {}
        losses['y'] = y_loss.data[0]
        for i, (regularizer, alpha) in enumerate(self.regularizers.values()):
            losses[i] = regularizer.loss(X, y_s, d).data[0]
        return losses

    def evaluate(self, loader, nb_batch=None, source=True):
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

        # evaluate regularizer
        for name, (regularizer, _) in iteritems(self.regularizers):
            for k, v in iteritems(regularizer._evaluate(loader, nb_batch, da_flag=True)):
                result['{}-{}'.format(name, k)] = v

        self.train()
        return result
