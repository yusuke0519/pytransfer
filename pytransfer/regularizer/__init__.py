# # -*- coding: utf-8 -*-
from collections import OrderedDict

import numpy as np
from sklearn import metrics
import torch
from torch import nn
from torch.utils import data


class _Reguralizer(nn.Module):
    def set_loader(self, dataset, batch_size):
        self.loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return self.loader

    def get_batch(self, as_variable=True, device='cpu'):
        assert self.loader is not None, "Please set loader before call this function"
        X, y, d = self.loader.__iter__().__next__()
        if as_variable:
            X = X.float().to(device)
            y = y.long().to(device)
            d = d.long().to(device)
        if hasattr(self.D, 'label_linear'):
            X = [X, y]
        return X, y, d

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def _evaluate(self, loader, nb_batch, da_flag=False):
        if nb_batch is None:
            nb_batch = len(loader)
        self.eval()
        targets = []
        preds = []
        loss = 0
        for i, (X, y, d) in enumerate(loader):
            with torch.no_grad():
                X = X.float().to(device)
                target = d.long().cuda()
                if not da_flag:
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