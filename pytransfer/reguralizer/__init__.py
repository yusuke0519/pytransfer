# # -*- coding: utf-8 -*-
from collections import OrderedDict

import numpy as np
from sklearn import metrics
from torch import nn
from torch.utils import data
from torch.autograd import Variable


class _Reguralizer(nn.Module):
    def set_learner(self, learner):
        self.learner = learner

    def set_loader(self, dataset, batch_size):
        self.loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return self.loader

    def get_batch(self, as_variable=True):
        assert self.loader is not None, "Please set loader before call this function"
        X, y, d = self.loader.__iter__().__next__()
        if as_variable:
            X = Variable(X.float().cuda())
            y = Variable(y.long().cuda())
            d = Variable(d.long().cuda())
        if hasattr(self.D, 'label_linear'):
            X = [X, y]
        return X, y, d

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

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
            pred = self(X)
            loss += self.loss(X, y, target).data[0]
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

class _DAReguralizer(_Reguralizer):
    def __init__(self, *args, **kwargs):
        super(_DAReguralizer, self).__init__(*args, **kwargs)
    
    def set_loader(self, source, target, batch_size):
        self.source_loader = data.DataLoader(source, batch_size=batch_size, shuffle=True)
        self.target_loader = data.DataLoader(target, batch_size=batch_size, shuffle=True)
        return self.source_loader, self.target_loader

    def get_batch(self, as_variable=True):
        assert self.source_loader is not None, "Please set loader before call this function"
        X_s, y_s, d_s = self.source_loader.__iter__().__next__()
        assert self.target_loader is not None, "Please set loader before call this function"
        X_t, _, d_t = self.target_loader.__iter__().__next__()
        if as_variable:
            X_s = Variable(X_s.float().cuda())
            y_s = Variable(y_s.long().cuda())
            X = Variable(torch.cat((X_s, X_t),0).float().cuda())
            d = Variable(torch.cat((d_s, (d_t + 1)), 0).long().cuda())# target label
        return X_s, y_s, X, d