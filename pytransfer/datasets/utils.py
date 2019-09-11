# # -*- coding: utf-8 -*-
import random

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler

from base import Subset


def del_key_in(start, stop, d):
    for k in np.arange(start, stop):
        if k in d.keys():
            del d[k]
    return d


def get_loader_for_domain(domain_indices, dataset, batch_size):
    indices = []
    for idx in domain_indices:
        if idx == 0:
            indices += range(0, dataset.cummulative_sizes[idx])
        else:
            indices += range(dataset.cummulative_sizes[idx-1], dataset.cummulative_sizes[idx])

    sampler = SubsetRandomSampler(indices)
    loader = data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return loader


def split_dataset(dataset, train_size=0.9):
    all_size = len(dataset)
    train_size = int(all_size * train_size)
    indices = range(all_size)
    random.shuffle(indices)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    return train_indices, test_indices


def get_joint_valid_dataloader(dataset, valid_size, batch_size):
    """ get joint domain validation dataset """
    train_indices, valid_indices = split_dataset(dataset, 1-valid_size)
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    valid_sampler = SubsetRandomSampler(valid_indices)
    valid_loader = data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    return train_loader, valid_loader


def get_disjoint_valid_dataloader(dataset, valid_domain_keys, batch_size):
    """ get disjoint domain validation dataset """
    valid_domain_indices = []
    for valid_domain_key in valid_domain_keys:
        valid_domain_indices.append(dataset.domain_dict[valid_domain_key])
    all_domain_indices = range(len(dataset.domain_keys))
    train_domain_indices = list(set(all_domain_indices) - set(valid_domain_indices))
    train_loader = get_loader_for_domain(train_domain_indices, dataset, batch_size)
    valid_loader = get_loader_for_domain(valid_domain_indices, dataset, batch_size)
    return train_loader, valid_loader


def prepare_loaders(left_out_key, validation, dataset_class, batch_size, random_state=123):
    if random_state is not None:
        random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed(random_state)

    if validation == 'test':
        # split test dataset into validation/test dataset. Two datasets comes from the same domain.
        train_dataset = dataset_class(dataset_class.get_disjoint_domains([left_out_key]))
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = dataset_class([left_out_key])
        test_loader, valid_loader = get_joint_valid_dataloader(test_dataset, 0.2, batch_size)
    elif validation == 'train':
        # split test dataset into train/validation dataset. Two datasets comes from the same domain.
        train_dataset = dataset_class(dataset_class.get_disjoint_domains([left_out_key]))
        train_loader, valid_loader = get_joint_valid_dataloader(train_dataset, 0.1, batch_size)
        test_dataset = dataset_class([left_out_key])
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    elif validation == 'disjoint':
        # split test dataset into train/validation dataset. Two datasets comes from different domains.
        train_dataset = dataset_class(dataset_class.get_disjoint_domains([left_out_key]))
        train_loader, valid_loader = get_disjoint_valid_dataloader(train_dataset,
                                                                   [random.choice(train_dataset.domain_keys)],
                                                                   batch_size)
        test_dataset = dataset_class([left_out_key])
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader, test_loader


def get_split_datasets(dataset, split_size):
    datasets1 = []
    datasets2 = []
    dataset_class = dataset.__class__
    domain_keys = dataset.domain_keys

    for domain_key in domain_keys:
        single_dataset = dataset.get_single_dataset(domain_key, **dataset.domain_specific_params())
        len_data = len(single_dataset)
        len_split = int(len_data * split_size)
        indices = torch.randperm(len_data)
        dataset1, dataset2 = Subset(single_dataset, indices[:len_split]), Subset(single_dataset, indices[len_split:])
        datasets1.append(dataset1)
        datasets2.append(dataset2)

    datasets1 = dataset_class(domain_keys=domain_keys, datasets=datasets1, **dataset.domain_specific_params())
    datasets2 = dataset_class(domain_keys=domain_keys, datasets=datasets2, **dataset.domain_specific_params())
    return datasets1, datasets2


def prepare_datasets(left_out_key, validation, dataset_class, require_domain, random_state=123):
    if random_state is not None:
        random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed(random_state)
    
    if isinstance(left_out_key, str):
        left_out_key = left_out_key.split(',')

    train_domain_keys = dataset_class.get_disjoint_domains(left_out_key)

    train_dataset = dataset_class(train_domain_keys, require_domain)
    if dataset_class.__name__ == 'Amazon':
        test_dataset = dataset_class(left_out_key, require_domain,
                                     vocabulary=train_dataset.vocabulary, vect=train_dataset.vect)
    else:
        test_dataset = dataset_class(left_out_key, require_domain)

    if validation == 'test':
        # split test dataset into validation/test dataset. Two datasets comes from the same domain.
        valid_dataset, test_dataset = get_split_datasets(test_dataset, 0.5)

    elif validation == 'train':
        # split test dataset into train/validation dataset. Two datasets comes from the same domain.
        train_dataset, valid_dataset = get_split_datasets(train_dataset, 0.8)

    elif validation == 'disjoint':
        # split test dataset into train/validation dataset. Two datasets comes from different domains.
        valid_domain_keys = [random.choice(train_domain_keys)]
        train_domain_keys = list(set(train_domain_keys) - set(valid_domain_keys))
        train_dataset = dataset_class(train_domain_keys, require_domain)
        valid_dataset = dataset_class(valid_domain_keys, require_domain)

    return train_dataset, valid_dataset, test_dataset


if __name__ == '__main__':
    from pytransfer.datasets.mnistr import MNISTR
    from IPython import embed
    left_out_key = MNISTR.all_domain_key[0]
    train_dataset, valid_dataset, test_dataset = prepare_datasets(left_out_key, "test", MNISTR, True)
    embed()
