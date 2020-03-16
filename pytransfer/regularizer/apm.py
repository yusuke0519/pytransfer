# # -*- coding: utf-8 -*-
import itertools
from collections import OrderedDict
import numpy as np
from sklearn import metrics
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from __init__ import DANReguralizer


class MovingAttributePerceptionLoss(DANReguralizer):
    """
    Calculating attribute perception loss

    Parameters
    ----------
    learner : instance of learner

    """
    def __init__(self, feature_extractor, D=None, discriminator_config=None, distance='KL', decay=0.7, K=1):
        super(MovingAttributePerceptionLoss, self).__init__(
            feature_extractor, discriminator_config=discriminator_config, K=K)
        self.distance = distance
        self.decay = decay

    def preprocess(self, dataset):
        self.initialize_centroids(dataset)

    def set_loader(self, dataset, batch_size):
        super(MovingAttributePerceptionLoss, self).set_loader(dataset, batch_size)
        self.preprocess(dataset)
        return self.loader

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

        centroids = np.zeros((num_domains, z.shape[1]))
        for j in range(num_domains):
            _filter = (d == j)
            new_centroid = z[_filter].mean(axis=0)
            centroids[_filter] = new_centroid
        self.centroids = centroids

    def update_centroids(self, X, y, d):
        g = self.perceptual(self.feature_extractor(X))
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

    def forward(self, z, y, d):
        g = self.perceptual(z)
        p_tgt = self.get_tgt_centroids(y, d, 'n')).float()
        g_log_softmax_repeat = torch.transpose(nn.functional.log_softmax(g, dim=-1).repeat(p_tgt.shape[1], 1, 1), 0, 1)
        return self.e_criterion(g_log_softmax_repeat, nn.functional.softmax(p_tgt, dim=-1)).sum(dim=-1).mean()

    @property
    def e_criterion(self):
        return nn.KLDivLoss(reduce=False)

    def loss(self, X, y, d):
        z = self.feature_extractorE(X)
        return self(z, y, d)

    def perceptual(self, z):
        return self.D.preactivation(z)

    def parameters(self):
        return self.D.parameters()

    def update(self, on_gpu=False):
        if self.stop_update:
            return None

        for _ in range(self.K):
            self.optimizer.zero_grad()
            X, _, d = self.get_batch(on_gpu)
            d_loss = self.d_loss(X, _, d)
            d_loss.backward()
            self.optimizer.step()

        X, y, d = self.get_batch(on_gpu)
        self.update_centroids(X, y, d)

    def validation_step(self, batch, batch_idx):
        X, y, d = batch
        z = self.feature_extractor(X)
        pred = self(z)
        loss = self.loss(z, y, d).item()
        d_hat = torch.argmax(pred, dim=1)
        acc = torch.sum(d == d_hat).item() / (len(d) * 1.0)
        d_mean = torch.exp(pred).mean(dim=0)
        d_ent_reg = torch.sum(-d_mean*torch.log(d_mean))

        return {'loss': loss, 'acc': acc, 'entropy': d_ent_reg}


class SemanticAlignedAttributePerceptionLoss(MovingAttributePerceptionLoss):
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
            centroids[_filter] = new_centroid
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
