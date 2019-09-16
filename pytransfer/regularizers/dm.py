# # -*- coding: utf-8 -*-
import itertools
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from . import _DiscriminatorBasedReg


def initialize_centroids(dataset, func=None, batch_size=4096):
    if func is None:
        func = lambda x: x

    z_list, y_list, d_list = [], [], []
    for X, y, d in data.DataLoader(dataset, batch_size=batch_size):
        z = func(Variable(X.float().cuda(), volatile=True))
        z_list.append(z.data.cpu().numpy())
        y_list.append(y.numpy())
        d_list.append(d.numpy())
    z = np.concatenate(z_list)
    y = np.concatenate(y_list)
    d = np.concatenate(d_list)

    num_domains = len(dataset.domain_keys)
    centroids = np.zeros((num_domains, z.shape[1]))
    for j in range(num_domains):
        _filter = (d == j)
        new_centroid = z[_filter].mean(axis=0)
        centroids[j] = new_centroid
    return centroids


class IIDM(_DiscriminatorBasedReg):
    """Calculating discriminator matching loss


    """
    def __init__(self, learner, D=None, discriminator_config=None, decay=0.7, K=1):
        super(IIDM, self).__init__(learner, discriminator_config=discriminator_config, K=K)
        self.decay = decay
        self.learner = learner
        self.e_criterion = nn.KLDivLoss(reduce=False)

    def set_loader(self, dataset, batch_size):
        super(IIDM, self).set_loader(dataset, batch_size)
        self.centroids = initialize_centroids(
            dataset, lambda x: self.D(self.learner.E(x)))
        num_classes = dataset.get('num_classes')
        num_domains = len(dataset.domain_keys)
        self.num_classes = num_classes
        self.num_domains = num_domains
        return self.loader

    def predict(self, X):
        z = self.learner.E(X)
        pred = self.D(z)
        return nn.LogSoftmax(dim=-1)(pred)

    def _D_loss(self, X, y, d):
        d_pred = self.predict(X)
        return self.criterion(d_pred, d)

    def update_centroids(self, X, y, d):
        g = self.D(self.learner.E(X))
        y = y.data.cpu()
        d = d.data.cpu()
        for j in np.unique(d):
            _filter = (d == j)
            if (_filter.numpy() == 1).sum() != 0:
                new_centroid = g.data.cpu().numpy()[_filter.numpy() == 1].mean(axis=0)
                self.centroids[j] = self.decay * self.centroids[j] + (1-self.decay) * new_centroid

    def get_tgt_centroids(self, y, d, filter_mode):
        y = y.data.cpu().numpy()
        d = d.data.cpu().numpy()
        num_domains, _ = self.centroids.shape
        if filter_mode == 'p':
            d_filter = lambda x: np.arange(num_domains) == x
        else:
            d_filter = lambda x: np.arange(num_domains) != x

        tgt_centroids = [None] * d.shape[0]
        for i in range(d.shape[0]):
            tgt_centroids[i] = [self.centroids[d_filter(d[i])]]
        tgt_centroids = np.concatenate(tgt_centroids)
        batch_size, _, g_size = tgt_centroids.shape
        return torch.from_numpy(tgt_centroids.reshape(batch_size, -1, g_size))

    def forward(self, X, y, d):
        z = self.learner.E(X)
        g = self.D(z)
        p_tgt = Variable(self.get_tgt_centroids(y, d, 'n')).float().cuda()
        g_log_softmax_repeat = torch.transpose(nn.functional.log_softmax(g, dim=-1).repeat(p_tgt.shape[1], 1, 1), 0, 1)
        return self.e_criterion(g_log_softmax_repeat, nn.functional.softmax(p_tgt, dim=-1)).sum(dim=-1).mean()

    def update(self):
        super(IIDM, self).update()
        X, y, d = self.get_batch()
        self.update_centroids(X, y, d)


class IIDMPlus(IIDM):
    """
    Calculating attribute perception loss

    Parameters
    ----------
    learner : instance of learner

    """
    def initialize_centroids(self, dataset, batch_size=4096):
        z_list, y_list, d_list = [], [], []
        for X, y, d in data.DataLoader(dataset, batch_size=batch_size):
            z = self.perceptual(self.learner.E(Variable(X.float().cuda(), volatile=True)))
            z_list.append(z.data.cpu().numpy())
            y_list.append(y.numpy())
            d_list.append(d.numpy())
        z = np.concatenate(z_list)
        y = np.concatenate(y_list)
        d = np.concatenate(d_list)

        num_classes = dataset.get('num_classes')
        num_domains = len(dataset.domain_keys)
        self.num_classes = num_classes
        self.num_domains = num_domains

        centroids = np.zeros((num_classes, num_domains, z.shape[1]))
        for i, j in itertools.product(range(num_classes), range(num_domains)):
            _filter = (y == i) & (d == j)
            new_centroid = z[_filter].mean(axis=0)
            centroids[i, j] = new_centroid
        self.centroids = centroids

    def update_centroids(self, X, y, d):
        g = self.perceptual(self.learner.E(X))
        y = y.data.cpu()
        d = d.data.cpu()
        for i, j in itertools.product(np.unique(y), np.unique(d)):
            _filter = (y == i) & (d == j)
            if (_filter.numpy() == 1).sum() != 0:
                new_centroid = g.data.cpu().numpy()[_filter.numpy() == 1].mean(axis=0)
                self.centroids[i, j] = self.decay * self.centroids[i, j] + (1-self.decay) * new_centroid

    def get_tgt_centroids(self, y, d, filter_mode):
        """Return target centroids with based on the touple of (y, d, filter_mode)

        Parameter
        ---------
        y : torch.LongTensor
            (N, ) size torch tensor where N is a batch size
        d : torch.LongTensor
            (N, ) size torch tensor where N is a batch size
        filter_mode : str
            This argument consists of two characters of 'p' or 'n'.
            The first character represents wheter it include positive class ('p') or negative class ('n'),
            and second character represents wheter it inlucd positive domain ('p') or negative domain ('n').

        Return
        ------
        tgt_centroids : np.array (TODO: should be torch tensor maybe?)
            (N, tgt_class*tgt_domain, perception_size) array, where N is a batch size,
            tgt_class and tgt_domain is the number of target class and domain respectively,
            and perception_size is the size of attribut perception vector.
        """

        y = y.data.cpu().numpy()
        d = d.data.cpu().numpy()
        num_classes, num_domains, _ = self.centroids.shape
        y_filter = lambda x: np.arange(num_classes) == x
        d_filter = lambda x: np.arange(num_domains) != x

        tgt_centroids = [None] * y.shape[0]
        for i in range(y.shape[0]):
            tgt_centroids[i] = [self.centroids[y_filter(y[i])][:, d_filter(d[i])]]
        tgt_centroids = np.concatenate(tgt_centroids)
        batch_size, _, _, g_size = tgt_centroids.shape
        return torch.from_numpy(tgt_centroids.reshape(batch_size, -1, g_size))
