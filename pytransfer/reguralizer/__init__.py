# # -*- coding: utf-8 -*-
import itertools
from future.utils import iteritems
import os
from collections import OrderedDict

import numpy as np
from sklearn import metrics
import torch
from torch import nn
from torch.utils import data
from torch.autograd import Variable

from dan import Discriminator
from pytransfer.datasets.utils import get_loader_for_domain
from pytransfer.learners.utils import get_classifier
from pytransfer.learners.utils import SpectralNorm


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


class DANReguralizer(_Reguralizer):
    """ 
    Domain advesarial reguralization for learning invariant representations.
    This is a special case of MultipleDAN (num_discriminator=1 and KL_weight=0).

    Reference
    ---------
    https://arxiv.org/abs/1505.07818
    http://papers.nips.cc/paper/6661-controllable-invariance-through-adversarial-feature-learning

    """

    def __init__(self, learner, D=None, discriminator_config=None, K=1):
        """
        Initialize dan base reguralizer

        Parameter
        ---------
        learner : instance of Learner
          TBA
        discriminator_config : dict
          configuration file for the discriminator
        K : int
          the # update of D in each iterations
        """
        super(DANReguralizer, self).__init__()

	self.stop_update = D is not None  # if D is shared with others, then not update here
	if D is None:
	    D = Discriminator(**discriminator_config)
        self.D = D.cuda()
        self.num_output = self.D.num_domains  # TODO: DANReguralizer should not assume that D has an attribute num_domain
        
        self.learner = learner
        self.K = K
        self.criterion = nn.NLLLoss()
        self.loader = None

    def forward(self, X):
        z = self.learner.E(X)
        return self.D(z)
    
    def loss(self, X, y, d):
        d_pred = self(X)
	d_loss = self.criterion(d_pred, d)
	return -1 * d_loss

    def d_loss(self, X, y, d):
        z = self.learner.E(X)
        d_pred = self.D(z)
	d_loss = self.criterion(d_pred, d)
        return d_loss

    def update(self):
	if self.stop_update:
	    return None

        for _ in range(self.K):
            self.optimizer.zero_grad()
            X, _, d = self.get_batch()
            d_loss = self.d_loss(X, _, d)
            d_loss.backward()
            self.optimizer.step()
    
    def parameters(self):
        return self.D.parameters()


class NSDANReguralizer(DANReguralizer):
    def loss(self, input, y, target):
        d_pred = self(input)
        index = Variable(torch.from_numpy(np.arange(len(target)))).cuda()
        loss = 0
        for i in np.unique(target.data.cpu()):
            tgt_idx = torch.masked_select(index, target!=i)
            loss += self.criterion(d_pred[tgt_idx], torch.ones_like(target)[tgt_idx] * i )
        return loss


class ConditionalEntropy(DANReguralizer):
    def loss(self, X, y, d):
	d_pred = self(X)
        d_mean = torch.exp(d_pred).mean(dim=0)
        d_ent_reg = torch.sum(-d_mean*torch.log(d_mean))
        return -1 * d_ent_reg


class MultilabelDAN(DANReguralizer):
    def __init__(self, learner, D=None, discriminator_config=None, K=1):
        """
        Initialize dan base reguralizer

        Parameter
        ---------
        learner : instance of Learner
          TBA
        discriminator_config : dict
          configuration file for the discriminator
        K : int
          the # update of D in each iterations
        """
        if discriminator_config is not None:
            discriminator_config['sn'] = True
            discriminator_config['bias'] = False
        super(MultilabelDAN, self).__init__(learner, D=D, discriminator_config=discriminator_config, K=K)

    def set_loader(self, dataset, batch_size):
        self.loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return self.loader

    def loss(self, X, y, d):
        G = self.get_G(X, d)
        factor = float(self.num_output)
        return (torch.trace(G) / factor) - (torch.sum(G) / (factor * factor))

    def get_G(self, X, d):
        index = Variable(torch.from_numpy(np.arange(len(d)))).cuda()
        d_preds = []
        for i in range(self.num_output):
            d_preds.append(self(X[torch.masked_select(index, d==i)]).mean(dim=0, keepdim=True))
        return torch.cat(d_preds)

    def parameters(self):
        return self.D.parameters()

    def _evaluate(self, loader, nb_batch):
        if nb_batch is None:
            nb_batch = len(loader)
        self.eval()
        targets = []
        preds = []
        loss = 0
        nb_valid_batch = 0
        for i, (X, y, d) in enumerate(loader):
            if len(np.unique(d.numpy())) != self.num_output:
                # TODO: this is ad-hock implementation to avoid the error
                # when the given loader only contain partial domain
                pass
            else:
                X = Variable(X.float().cuda(), volatile=True)
                target = Variable(d.long().cuda(), volatile=True)
                loss += self.loss(X, y, target).data[0]
                nb_valid_batch += 1
            if i+1 == nb_batch:
                break
        loss /= nb_batch

        result = OrderedDict()
        result['accuracy'] = np.nan
        result['f1macro'] = np.nan
        result['loss'] = loss
        self.train()
        return result


class PairwiseWassersteinDistance(DANReguralizer):
    def __init__(self, learner, D=None, discriminator_config=None, K=1):
        """
        Initialize dan base reguralizer

        Parameter
        ---------
        learner : instance of Learner
          TBA
        discriminator_config : dict
          configuration file for the discriminator
        K : int
          the # update of D in each iterations
        """
        if discriminator_config is not None:
            discriminator_config['sn'] = True
            discriminator_config['bias'] = False
        super(PairwiseWassersteinDistance, self).__init__(learner, D=D, discriminator_config=discriminator_config, K=K)
        self.criterion = None

    def set_loader(self, dataset, batch_size):
        self.loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return self.loader

    def loss(self, X, y, d):
        G = self.get_G(X, d)
        factor = float(self.num_output)
        return (G.diag() - G).abs().sum() / (factor * (factor-1))

    def get_G(self, X, d):
        index = Variable(torch.from_numpy(np.arange(len(d)))).cuda()
        d_preds = []
        # for i in range(self.num_output):
        #     d_preds.append(self(X[torch.masked_select(index, d==i)]).mean(dim=0, keepdim=True))
        g = self(X)
        for i in range(g.shape[1]):
            tgt_index = torch.masked_select(index, d==i)
            if len(tgt_index) == 0:
                d_preds.append(torch.zeros_like(g.mean(dim=0, keepdim=True)))
            else:
                d_preds.append(g[torch.masked_select(index, d==i)].mean(dim=0, keepdim=True))
        return torch.cat(d_preds)

    def parameters(self):
        return self.D.parameters()

    def _evaluate(self, loader, nb_batch):
        if nb_batch is None:
            nb_batch = len(loader)
        self.eval()
        targets = []
        preds = []
        loss = 0
        nb_valid_batch = 0
        for i, (X, y, d) in enumerate(loader):
            if len(np.unique(d.numpy())) != self.num_output:
                # TODO: this is ad-hock implementation to avoid the error
                # when the given loader only contain partial domain
                pass
            else:
                X = Variable(X.float().cuda(), volatile=True)
                target = Variable(d.long().cuda(), volatile=True)
                loss += self.loss(X, y, target).data[0]
                nb_valid_batch += 1
            if i+1 == nb_batch:
                break
        loss /= nb_batch

        result = OrderedDict()
        result['accuracy'] = np.nan
        result['f1macro'] = np.nan
        result['loss'] = loss
        self.train()
        return result
    
    def d_loss(self, X, y, d):
        return -1 * self.loss(X, y, d)


class OnesidePWD(PairwiseWassersteinDistance):
    def loss(self, X, y, d):
        G = self.get_G(X, d)
        factor = float(self.num_output)
        G_copy = Variable(G.data)  # stop gradient
        return (G.diag() - G_copy).abs().sum() / (factor * (factor-1))


class SlicedWasserstein(_Reguralizer):
    def __init__(self, learner, num_projection=50, p=2, discriminator_config=None):
        super(SlicedWasserstein, self).__init__()
        self.learner = learner
        self.num_projection = num_projection
        self.p = p
        self.loader = None
    
    def forward(self, z, d):
        index = Variable(torch.from_numpy(np.arange(len(d)))).cuda()
        z_list = []
        for i in np.unique(d.data.cpu()):
            z_list.append(z[torch.masked_select(index, d == i)])
        loss = 0
        num_combination = 0
        for z1, z2 in itertools.combinations(z_list, 2):
            loss += sliced_wasserstein_distance(z1, z2, num_projections=self.num_projection, p=self.p)
            num_combination += 1
        return loss / num_combination

    def loss(self, X, y, d):
        z = self.learner.E(X)
        return self(z, d)

    def update(self):
        pass

    def _evaluate(self, loader, nb_batch):
        if nb_batch is None:
            nb_batch = len(loader)
        self.eval()
        targets = []
        preds = []
        loss = 0
        nb_valid_batch = 0
        for i, (X, y, d) in enumerate(loader):
            if len(np.unique(d.numpy())) <= 1:
                continue
            X = Variable(X.float().cuda(), volatile=True)
            target = Variable(d.long().cuda(), volatile=True)
            loss += self.loss(X, y, target).data[0]
            nb_valid_batch += 1
            if i+1 == nb_batch:
                break
        loss /= nb_batch

        result = OrderedDict()
        result['loss'] = loss
        self.train()
        return result


class MinMaxSlicedWasserstein(SlicedWasserstein):
    def __init__(self, learner, num_projection=50, p=2, discriminator_config=None, K=1):
        super(MinMaxSlicedWasserstein, self).__init__(learner, num_projection, p)
        if discriminator_config is not None:
            discriminator_config['sn'] = True
            self.D = get_classifier(**discriminator_config).cuda()
        self.K = K
    
    def parameters(self):
        return self.D.parameters()
    
    def loss(self, X, y, d):
        z = self.learner.E(X)
        z = self.D(z)
        return self(z, d)
    
    def update(self):
        for _ in range(self.K):
            self.optimizer.zero_grad()
            X, _, d = self.get_batch()
            d_loss = -1 * self.loss(X, _, d)
            d_loss.backward()
            self.optimizer.step()


class DiscriminativeSlicedWasserstein(MinMaxSlicedWasserstein):
    def __init__(self, learner, num_domain, num_projection=50, p=2, discriminator_config=None, K=1):
        super(DiscriminativeSlicedWasserstein, self).__init__(learner, num_projection, p, discriminator_config, K)
        hiddens = discriminator_config['layers']
        module = nn.Linear(hiddens[-1], num_domain)
        module = SpectralNorm(module)
        self.linear = module.cuda()
        self.activation = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss()
    
    def parameters(self):
        return list(self.D.parameters()) + list(self.linear.parameters())
    
    def update(self):
        for _ in range(self.K):
            self.optimizer.zero_grad()
            X, _, d = self.get_batch()
            z = self.learner.E(X)
            z = self.D(z)
            z = self.linear(z)
            d_pred = self.activation(z)
            d_loss = self.criterion(d_pred, d)
            d_loss.backward()
            self.optimizer.step()

    def _evaluate(self, loader, nb_batch):
        if nb_batch is None:
            nb_batch = len(loader)
        self.eval()
        targets = []
        preds = []
        loss = 0
        nb_valid_batch = 0
        for i, (X, y, d) in enumerate(loader):
            if len(np.unique(d.numpy())) <= 1:
                continue
            X = Variable(X.float().cuda(), volatile=True)
            target = Variable(d.long().cuda(), volatile=True)
            loss += self.loss(X, y, target).data[0]
            nb_valid_batch += 1
            z = self.learner.E(X)
            z = self.D(z)
            z = self.linear(z)
            pred = self.activation(z)
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
        print(result)
        return result


def sliced_wasserstein_distance(z1, z2, num_projections=50, p=2):
    """ Sliced Wasserstein Distance between encoded samples and drawn distribution samples.
        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples
            distribution_samples (torch.Tensor): tensor of drawn distribution training samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
            device (torch.device): torch device (default 'cpu')
        Return:
            torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
    """
    # equalize the # samples
    min_size = min(z1.shape[0], z2.shape[0])
    z1 = z1[:min_size]
    z2 = z2[:min_size]
    # derive latent space dimension size from random samples drawn from latent prior distribution
    embedding_dim = z1.size(1)
    # generate random projections in latent space
    projections = rand_projections(embedding_dim, num_projections).cuda()
    # calculate projections through the encoded samples
    encoded_projections = z2.matmul(projections.transpose(0, 1))
    # calculate projections through the prior distribution random samples
    distribution_projections = (z1.matmul(projections.transpose(0, 1)))
    # calculate the sliced wasserstein distance by
    # sorting the samples per random projection and
    # calculating the difference between the
    # encoded samples and drawn random samples
    # per random projection
    wasserstein_distance = (torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                            torch.sort(distribution_projections.transpose(0, 1), dim=1)[0])
    # distance between latent space prior and encoded distributions
    # power of 2 by default for Wasserstein-2
    wasserstein_distance = torch.pow(wasserstein_distance, p)
    # approximate mean wasserstein_distance for each projection
    return wasserstein_distance.mean()


def rand_projections(embedding_dim, num_samples=50):
    """This function generates `num_samples` random samples from the latent space's unit sphere.
        Args:
            embedding_dim (int): embedding dimensionality
            num_samples (int): number of random projection samples
        Return:
            torch.Tensor: tensor of size (num_samples, embedding_dim)
    """
    projections = [w / np.sqrt((w**2).sum())  # L2 normalization
                   for w in np.random.normal(size=(num_samples, embedding_dim))]
    projections = np.asarray(projections)
    return Variable(torch.from_numpy(projections).type(torch.FloatTensor))


class MultipleDAN(_Reguralizer):
    def __init__(self, learner, num_discriminator, discriminator_config=None, K=1, KL_weight=0.0):
        """
        Initialize dan base reguralizer

        Parameter
        ---------
        learner : instance of Learner
          TBA
        num_discriminator : int
          # discrimnator for ensembling
        discriminator_config : dict
          configuration file for the discriminator
        K : int
          the # update of D in each iterations
        KL_weight : float
          the weight parameter for KL divergence
        """

        super(MultipleDAN, self).__init__()
    
        discriminator_config['use_softmax'] = False
	self.D = [Discriminator(**discriminator_config).cuda() for i in range(num_discriminator)]
        self.num_output = self.D[0].num_domains
        
        self.learner = learner
        self.K = K
        self.criterion = nn.NLLLoss()
        self.loader = None
        self.KL_weight = KL_weight

    def forward(self, X):
        """
        Return (batch_size, num_output, num_discriminator) tensor

        """
        z = self.learner.E(X)
        g = torch.stack([_D(z) for _D in self.D], dim=-1)
        return nn.functional.log_softmax(g.mean(dim=-1), dim=1)
    
    def loss(self, X, y, d):
        d_pred = self(X)
        return -1 * (nn.NLLLoss()(d_pred, d))

    def update(self):
        """
        Train all discriminator with the same data point

        TODO: Better design 
        """
        for _ in range(self.K):
            self.optimizer.zero_grad()
            X, _, d = self.get_batch()
            z = Variable(self.learner.E(X).data)
            loss = 0
            mean_prob = torch.exp(Variable(self(X).data))
            for _D in self.D:
                d_pred = nn.functional.log_softmax(_D(z), dim=1)
                loss += nn.NLLLoss()(d_pred, d)
                kl_loss = nn.KLDivLoss()(d_pred, mean_prob)
                loss -= self.KL_weight * kl_loss
            loss.backward()
            self.optimizer.step()

    def parameters(self):
        weights = []
        for _D in self.D:
            weights += _D.parameters()

        return weights


class AttributePerceptionLoss(_Reguralizer):
    """
    Calculating attribute perception loss

    Parameters
    ----------
    learner : instance of learner

    """
    def __init__(self, learner, D=None, discriminator_config=None, distance='KL', decay=0.7):
        super(AttributePerceptionLoss, self).__init__()
    
    def preprocess(self, dataset):
        self.initialize_centroids(dataset)

    def initialize_centroids(self, dataset, batch_size=4096):
        z_list, y_list, d_list = [], [], []
        for X, y, d in data.DataLoader(dataset, batch_size=batch_size):
            z = self.perceptual(self.E(Variable(X.float().cuda(), volatile=True)))
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
        g = self.perceptual(self.E(Variable(X.float().cuda(), volatile=True)))
        for i, j in itertools.product(np.unique(y), np.unique(d)):
            _filter = (y == i) & (d == j)
            if (_filter.numpy()==1).sum() != 0:
                new_centroid = g.data.cpu().numpy()[_filter.numpy()==1].mean(axis=0)
                self.centroids[i, j] = self.decay * self.centroids[i, j] + (1-self.decay) * new_centroid

    def perceptual(self, z):
        return self.D.preactivation(z)        
