# # -*- coding: utf-8 -*-
import itertools
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch


class _CatClassifierBase(nn.Module):
    def __init__(self, D, optimizer, optimizer_params=None):
        super(_CatClassifierBase, self).__init__()
        if optimizer_params is None:
            optimizer_params = {}
        self.D = D
        self.optimizer = optimizer(D.parameters(), **optimizer_params)
        self.d_criterion = nn.NLLLoss()

    def forward(self, input, target):
        """
        Return approximated invariance measurements (lower is better invariant)

        Parameter
        ---------
        input : input for the Discriminator $D$
        target : true label for each $N$ input
        """
        raise NotImplementedError

    def train(self, input, target):
        self.optimizer.zero_grad()
        loss = self.d_loss(input, target)
        loss.backward()
        self.optimizer.step()

    def d_loss(self, input, target):
        d_pred = self.D(input)
        return self.d_criterion(d_pred, target)

    @property
    def e_criterion(self):
        raise NotImplementedError

    def e_loss(self, input, target):
        raise NotImplementedError


class AdversarialCatClassifier(_CatClassifierBase):
    """
    Adversarial categorical loss for learning attribute invariant representations.
    Specifically, train a categorical attribute classifier $D$ first, and regurn a log-likelihood of $D$ given a z, a touple

    Reference
    ---------
    https://arxiv.org/abs/1505.07818
    http://papers.nips.cc/paper/6661-controllable-invariance-through-adversarial-feature-learning
    """
    def forward(self, input, target):
        return self.e_loss(input, target)

    @property
    def e_criterion(self):
        return nn.NLLLoss()

    def e_loss(self, input, target):
        return -1 * self.e_criterion(self.D(input), target)


def kl_mean_sample(tgt_mean, sample):
    """

    Parameter
    ---------
    tgt_mean : (num_target, dim_g)
    sample : (batch_size, dim_g)

    """
    sample_repeat = torch.transpose(
        nn.functional.log_softmax(sample, dim=-1).repeat(tgt_mean.shape[1], 1, 1), 0, 1)
    return nn.KLDivLoss(reduce=False)(sample_repeat, nn.functional.softmax(tgt_mean, dim=-1)).sum(dim=-1).mean()


def L2_mean_sample(tgt_mean, sample):
    """

    Parameter
    ---------
    tgt_mean : (num_target, dim_g)
    sample : (batch_size, dim_g)

    """
    sample_repeat = torch.transpose(nn.functional.repeat(tgt_mean.shape[1], 1, 1), 0, 1)
    return nn.MSELoss(reduce=False)(sample_repeat, tgt_mean).mean()


class AttributePerceptionLoss(_CatClassifierBase):
    """
    Attribue perception loss, which aligns the attribute perception (hidden activations) between different attribute

    """
    def __init__(self, D, optimizer, optimizer_params=None, decay=1.0, distance='KL'):
        super(AttributePerceptionLoss, self).__init__(D, optimizer, optimizer_params)
        self.decay = decay
        self.distance = distance

    def init_centroids(self, z, y, d):
        g = self.D.preactivation(Variable(z.float()).cuda())
        g = g.data.cpu().numpy()
        num_classes = len(np.unique(y))
        num_domains = len(np.unique(d))

        centroids = np.zeros((num_classes, num_domains, g.shape[1]))
        for i, j in itertools.product(range(num_classes), range(num_domains)):
            _filter = (y == i) & (d == j)
            centroids[i, j] = g[_filter].mean(axis=0)
        self.centroids = centroids
        return centroids

    def update_centroids(self, z, y, d):
        g = self.D.preactivation(z)
        g = g.data.cpu().numpy()
        for i, j in itertools.product(np.unique(y), np.unique(d)):
            _filter = (y == i) & (d == j)
            if (_filter == 1).sum() != 0:
                new_centroid = g[_filter == 1].mean(axis=0)
                self.centroids[i, j] = self.decay * self.centroids[i, j] + (1-self.decay) * new_centroid
        return self.centroids

    def forward(self, input, target):
        return self.e_loss(input, target)

    @property
    def e_criterion(self):
        return nn.KLDivLoss(reduce=False)

    def e_loss(self, input, target):
        z, y = input
        g = self.D.preactivation(z)

        if self.distance == 'KL':
            p_tgt = Variable(self.get_tgt_centroids(y, target, 'pn')).float().cuda()
            return kl_mean_sample(p_tgt, g)

        else:
            raise Exception("Not implemeted error")

    def get_tgt_centroids(self, y, d, filter_mode):
        """
        Return target centroids with based on the touple of (y, d, filter_mode)

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

        y = y.numpy()
        d = d.numpy()
        num_classes, num_domains, _ = self.centroids.shape
        if filter_mode[0] == 'p':
            y_filter = lambda x: np.arange(num_classes) == x
        else:
            y_filter = lambda x: np.arange(num_classes) != x

        if filter_mode[1] == 'p':
            d_filter = lambda x: np.arange(num_domains) == x
        else:
            d_filter = lambda x: np.arange(num_domains) != x
        tgt_centroids = [None] * y.shape[0]

        for i in range(y.shape[0]):
            tgt_centroids[i] = [self.centroids[y_filter(y[i])][:, d_filter(d[i])]]
        tgt_centroids = np.concatenate(tgt_centroids)
        batch_size, _, _, g_size = tgt_centroids.shape
        return torch.from_numpy(tgt_centroids.reshape(batch_size, -1, g_size))


class NSAdversarialCatClassifier(_CatClassifierBase):
    def forward(self, input, target):
        return self.e_loss(input, target)

    @property
    def e_criterion(self):
        return nn.NLLLoss()

    def e_loss(self, input, target):
        d_pred = self.D(input)
        loss = 0
        index = Variable(torch.from_numpy(np.arange(len(target)))).cuda()

        for i in np.unique(target.data.cpu()):
            tgt_idx = torch.masked_select(index, target!=i)
            loss += self.e_criterion(d_pred[tgt_idx], torch.ones_like(target)[tgt_idx] * i )
        return loss


class _IPMBase(nn.Module):
    """
    Base class for computing measure ments of IMP familiy (e.g., wasserstein, )
    
    This class include any divergence measurements conform to the following equations:
    $ min{sup_{f\in\mathcal{F}_{nn}} V(E, D):= \E[\phi(f(z1))] - \E[\phi(f(z1))]}. 
    
    """
    def __init__(self, D, optimizer, optimizer_params=None):
        super(_IPMBase, self).__init__()
        if optimizer_params is None:
            optimizer_params = {}
        self.D = D
        self.optimizer = optimizer(D.parameters(), **optimizer_params)
        
    def forward(self, input, target):
        """
        Return approximated invariance measurements (lower is better invariant)
        
        
        Parameter
        ---------
        input : input for the Discriminator $D$ 
        target : true label for each $N$ input
        """
        
        raise NotImplementedError
    
    def train(self, input, target):
        self.optimizer.zero_grad()
        loss = -1 * self(input, target)
        loss.backward()
        self.optimizer.step()

        
class PairWiseWasserstein(_IPMBase):
    """
    Pair-wise wasserstein distance (JSAI2019)
    
    """
    def forward(self, input, target):
        G = self.get_G(input, target)
        factor = float(G.shape[0])
        # G_copy = Variable(G.data)
        # self.G = G
        return (G.diag() - G).abs().sum() / (factor*(factor-1))
        # return (torch.trace(G) / factor) - (torch.sum(G) / (factor * factor))

    def get_G(self, z, d):
        """
        Create num_domain, num_domain matrix, where each element i, j represetnts
        $ V(E, D):= \E_{domain i}[\phi(f(z))] - \E_{domain j}[\phi(f(z))]. 
        
        """
        index = Variable(torch.from_numpy(np.arange(len(d)))).cuda()
        d_preds = []
        g = self.D(z)
        
        for i in range(g.shape[1]):
            tgt_index = torch.masked_select(index, d==i)
            if len(tgt_index) == 0:
                d_preds.append(torch.zeros_like(g.mean(dim=0, keepdim=True)))
            else:
                d_preds.append(g[torch.masked_select(index, d==i)].mean(dim=0, keepdim=True))
        return torch.cat(d_preds)
        # return torch.cat([torch.zeros_like(g.mean(dim=0, keepdim=True)) if (torch.masked_select(index, d==i) == 0) else g[torch.masked_select(index, d==i)].mean(dim=0, keepdim=True) for i in range(g.shape[1])])




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


class SlicedWasserstein(nn.Module):

    def __init__(self, num_projection=50, p=2):
        super(SlicedWasserstein, self).__init__()
        self.num_projection=50
        self.p = 2

    def forward(self, z, d):
        index = Variable(torch.from_numpy(np.arange(len(d)))).cuda()
        z_list = []
        for i in np.unique(d.data.cpu()):
            z_list.append(z[torch.masked_select(index, d==i)])
        loss = 0
        num_combination = 0
        for z1, z2 in itertools.combinations(z_list, 2):
            loss += sliced_wasserstein_distance(z1, z2)
            num_combination += 1
        return loss / num_combination

    def train(self, z, d):
        pass


class DiscriminativeSlicedWasserstein(nn.Module):

    def __init__(self, D, optimizer, optimizer_params=None, num_projection=50, p=2):
        super(DiscriminativeSlicedWasserstein, self).__init__()
        self.num_projection = 50
        self.p = 2
        if optimizer_params is None:
            optimizer_params = {}
        self.D = D
        self.optimizer = optimizer(D.parameters(), **optimizer_params)
        self.activation = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss()

    def train(self, z, d):
        self.optimizer.zero_grad()
        d_pred = self.activation(self.D(z))
        loss = self.criterion(d_pred, d)
        loss.backward()
        self.optimizer.step()

    def forward(self, z, d):
        z = self.D(z)
        index = Variable(torch.from_numpy(np.arange(len(d)))).cuda()
        z_list = []
        for i in np.unique(d.data.cpu()):
            z_list.append(z[torch.masked_select(index, d==i)])
        loss = 0
        num_combination = 0
        for z1, z2 in itertools.combinations(z_list, 2):
            loss += sliced_wasserstein_distance(z1, z2)
            num_combination += 1
        return loss / num_combination
