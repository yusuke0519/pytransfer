# -*- coding: utf-8 -*-
import pickle
import os
import gzip
import wget

import numpy as np
import torch.utils.data as data

from .base import DomainDatasetBase


CONFIG = {}
CONFIG['url'] = 'https://github.com/ghif/mtae/raw/master/MNIST_6rot.pkl.gz'
CONFIG['out'] = os.path.expanduser('~/.torch/datasets')


class _SingleMNISTR(data.Dataset):
    path = os.path.expanduser('~/.torch/datasets/MNIST_6rot.pkl.gz')
    all_domain_key = ['M0', 'M15', 'M30', 'M45', 'M60', 'M75']
    input_shape = (1, 16, 16)
    num_classes = 10

    def __init__(self, domain_key):
        assert domain_key in self.all_domain_key
        if not os.path.exists(self.path):
            self.download()
        self.domain_key = domain_key
        domain_id = self.all_domain_key.index(domain_key)
        img_rows, img_cols = self.input_shape[1:]

        all_domains = pickle.load(gzip.open(self.path, 'rb'), encoding='latin1')
        X, y = all_domains[domain_id]
        X = X.reshape(X.shape[0], 1, img_rows, img_cols)
        self.X = X
        self.y = y

    def download(self):
        output_dir = os.path.dirname(self.path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        wget.download(CONFIG['url'], out=self.path)

    def __getitem__(self, index):
        return self.X[index], self.y[index], self.domain_key

    def __len__(self):
        return len(self.y)


class MNISTR(DomainDatasetBase):
    """ Rotation version MNIST

    Args:
      domain_keys: a list of domains

    """
    SingleDataset = _SingleMNISTR


class _SingleBiasedMNISTR(data.Dataset):
    """
        | domain_key | what percentage of 1~5 are used |
        | 0          | 100% |
        | 15         | 85%  |
        | 30         | 70%  |
        | 45         | 55%  |
        | 60         | 40%  |
        | 75         | 25%  |
    """
    path = os.path.expanduser('~/.torch/datasets/MNIST_6rot.pkl.gz')
    all_domain_key = ['0', '15', '30', '45', '60', '75']
    input_shape = (1, 16, 16)
    num_classes = 10
    bias = {'0' : 1,
            '15': 0.85,
            '30': 0.7,
            '45': 0.55,
            '60': 0.4,
            '75': 0.25}

    def __init__(self, domain_key):
        assert domain_key in self.all_domain_key
        if not os.path.exists(self.path):
            self.download()
        self.domain_key = domain_key
        domain_id = self.all_domain_key.index(domain_key)
        img_rows, img_cols = self.input_shape[1:]

        all_domains = pickle.load(gzip.open(self.path, 'rb'))
        X, y = all_domains[domain_id]
        X = X.reshape(X.shape[0], 1, img_rows, img_cols)
        indices = self.get_biased_indices(y)
        self.X = X[indices]
        self.y = y[indices]

    def download(self):
        output_dir = os.path.dirname(self.path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        wget.download(CONFIG['url'], out=self.path)

    def __getitem__(self, index):
        return self.X[index], self.y[index], self.domain_key

    def __len__(self):
        return len(self.y)

    def get_biased_indices(self, y):
        def trim(indices):
            biased_len = int(len(indices) * self.bias[self.domain_key])
            indices = indices[:biased_len]
            return indices

        zero_to_four = []
        for i in range(5):
            zero_to_four += trim(np.where(y == i)[0]).tolist()
        five_to_nine = np.where(y >= 5)[0]
        if len(zero_to_four) == 0:
            return five_to_nine
        indices = np.append(zero_to_four, five_to_nine)
        return indices


class BiasedMNISTR(DomainDatasetBase):
    """ Rotation version MNIST with d -> y

    Args:
      domain_keys: a list of domains

    """
    SingleDataset = _SingleBiasedMNISTR


def get_biased_mnistr(bias):
    """
    Args:
        bias: dict of class and domain relationship.
              example:{'0' : 1,
                      '15': 0.85,
                      '30': 0.7,
                      '45': 0.55,
                      '60': 0.40,
                      '75': 0.25}
    """
    _SingleBiasedMNISTR.bias = bias
    BiasedMNISTR.SingleDataset = _SingleBiasedMNISTR
    return BiasedMNISTR


if __name__ == '__main__':
    dataset = MNISTR(domain_keys=['0', '30', '60'])
    dataset = MNISTR(domain_keys='0,15,30')
    dataset = MNISTR(domain_keys=MNISTR.get_disjoint_domains(['0', '30', '60']))
    print(len(np.where(dataset.datasets[0].y < 5)[0]))
    print(len(np.where(dataset.datasets[0].y >= 5)[0]))

    dataset = BiasedMNISTR(domain_keys=['15', '30', '45'])
    print(len(np.where(dataset.datasets[0].y < 5)[0]))
    print(len(np.where(dataset.datasets[0].y >= 5)[0]))

    UnnaturalMNISTR = get_biased_mnistr(
        {'0' : 1,
        '15': 0.85,
        '30': 0.7,
        '45': 0.55,
        '60': 1,
        '75': 0.25})
    dataset = UnnaturalMNISTR(domain_keys=['60', '75'])
    print(len(np.where(dataset.datasets[0].y < 5)[0]))
    print(len(np.where(dataset.datasets[0].y >= 5)[0]))
    print(len(np.where(dataset.datasets[1].y < 5)[0]))
    print(len(np.where(dataset.datasets[1].y >= 5)[0]))
