# # -*- coding: utf-8 -*-
import torch
import torch.utils.data as data


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class DomainDatasetBase(data.ConcatDataset):
    """ Base class for multi domain dataset class
    subclasses must have these class variables:
        all_domain_key: domain keys for the dataset
        SingleDataset: dataset class for one domain dataset
    """
    SingleDataset = None

    def __init__(self, domain_keys, require_domain=True, datasets=None):
        """ Base class for multi domain dataset class
        Args:
            domain_keys: list or str
            require_domain: Boolean
        """
        assert isinstance(domain_keys, list) or isinstance(domain_keys, str)
        if isinstance(domain_keys, list):
            self.domain_keys = domain_keys
        elif isinstance(domain_keys, str):
            self.domain_keys = [x for x in domain_keys.split(',')]
        self.require_domain = require_domain
        self.domain_dict = dict(zip(self.domain_keys, range(len(self.domain_keys))))

        if datasets is None:
            datasets = []
            for domain_key in self.domain_keys:
                datasets += [self.get_single_dataset(domain_key, **self.domain_specific_params())]
        super(DomainDatasetBase, self).__init__(datasets)

    def get_single_dataset(self, domain_key, **kwargs):
        return self.SingleDataset(domain_key, **kwargs)

    def domain_specific_params(self):
        return {}

    def __getitem__(self, idx):
        X, y, d = super(DomainDatasetBase, self).__getitem__(idx)
        if self.require_domain:
            D = self.domain_dict[d]
            return X, y, D
        return X, y

    @classmethod
    def get_disjoint_domains(cls, domain_keys):
        if isinstance(domain_keys, str):
            domain_keys = [x for x in domain_keys.split(',')]

        all_domain_keys = cls.get('all_domain_key')[:]
        for domain_key in domain_keys:
            all_domain_keys.remove(domain_key)
        return all_domain_keys

    @classmethod
    def get(cls, name):
        if hasattr(cls, name):
            return getattr(cls, name)
        elif hasattr(cls.SingleDataset, name):
            return getattr(cls.SingleDataset, name)
