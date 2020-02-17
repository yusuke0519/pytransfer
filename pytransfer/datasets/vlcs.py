# -*- coding: utf-8 -*-

import os
import wget
from scipy.io import loadmat

import torch.utils.data as data

from .base import DomainDatasetBase

CONFIG = {}
CONFIG['V'] = 'https://drive.google.com/uc?export=download&id=1bwuegzRa3MONhmjlJnf6VjWwK1sEkXER'
CONFIG['L'] = 'https://drive.google.com/uc?export=download&id=1kaVCI7o9SW4U8UUmc_M4OCqwtBEsNVFJ'
CONFIG['C'] = 'https://drive.google.com/uc?export=download&id=1w4XAUGkipynlb1XTErcJA8HE8AwppLiX'
CONFIG['S'] = 'https://drive.google.com/uc?export=download&id=19FT_9YSF2_EYudWEjOXPRNQDQUO4uRft'
CONFIG['out'] = os.path.expanduser('~/.torch/datasets')


class _SingleVLCS(data.Dataset):
    path = os.path.expanduser('~/.torch/datasets/')
    all_domain_key = ['V', 'L', 'C', 'S']
    input_shape = (4096,)
    num_classes = 5

    def __init__(self, domain_key):
        assert domain_key in self.all_domain_key
        domain_path = self.path + domain_key + '.mat'
        if not os.path.exists(domain_path):
            self.download(domain_path, domain_key)
        self.domain_key = domain_key
        data = loadmat(domain_path, squeeze_me=True)['data']
        self.data = data

    def download(self, domain_path, domain_key):
        output_dir = os.path.dirname(domain_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        wget.download(CONFIG[domain_key], out=domain_path)

    def __getitem__(self, index):
        return self.data[index][:self.input_shape[0]], self.data[index][self.input_shape[0]] - 1, self.domain_key

    def __len__(self):
        return len(self.data)


class VLCS(DomainDatasetBase):
    """ VLCS

    Args:
      domain_keys: a list of domains

    """
    SingleDataset = _SingleVLCS
