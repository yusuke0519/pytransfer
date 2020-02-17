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
from dan import Discriminator


class MovingAttributePerceptionLoss(DANReguralizer):
    """
    Calculating attribute perception loss

    Parameters
    ----------
    learner : instance of learner

    """
    def __init__(self, learner, D=None, discriminator_config=None, distance='KL', decay=0.7, K=1):
        super(MovingAttributePerceptionLoss, self).__init__(learner, discriminator_config=discriminator_config, K=K)
        self.distance = distance
        self.decay = decay
        self.learner = learner

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
        g = self.perceptual(self.learner.E(X))
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
        p_tgt = Variable(self.get_tgt_centroids(y, d, 'n')).float().cuda()
        g_log_softmax_repeat = torch.transpose(nn.functional.log_softmax(g, dim=-1).repeat(p_tgt.shape[1], 1, 1), 0, 1)
        return self.e_criterion(g_log_softmax_repeat, nn.functional.softmax(p_tgt, dim=-1)).sum(dim=-1).mean()

    @property
    def e_criterion(self):
        return nn.KLDivLoss(reduce=False)

    def loss(self, X, y, d):
        z = self.learner.E(X)
        return self(z, y, d)

    def perceptual(self, z):
        return self.D.preactivation(z)
    def parameters(self):
        return self.D.parameters()
    
    def _evaluate(self, loader, nb_batch):
        if nb_batch is None:
            nb_batch = len(loader)
        self.eval()
        targets = []
        preds = []
        loss = 0
        for i, (X, y, d) in enumerate(loader):
            X = Variable(X.float().cuda(), volatile=True)
            y_target = Variable(y.long().cuda(), volatile=True)
            target = Variable(d.long().cuda(), volatile=True)
            if len(np.unique(target.data.cpu())) <= 1:
                continue
            z = self.learner.E(X)
            pred = self.D(z)
            loss += self.loss(X, y_target, target).data[0]
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
    
    def update(self):
	if self.stop_update:
	    return None

        for _ in range(self.K):
            self.optimizer.zero_grad()
            X, _, d = self.get_batch()
            d_loss = self.d_loss(X, _, d)
            d_loss.backward()
            self.optimizer.step()
    
        X, y, d = self.get_batch()
        self.update_centroids(X, y, d)


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
