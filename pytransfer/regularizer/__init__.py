# # -*- coding: utf-8 -*-

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

