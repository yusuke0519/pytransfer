# -*- coding: utf-8 -*-

import os
import wget
import shutil
import tarfile

import numpy as np
import pandas as pd
import torch.utils.data as data

from .base import DomainDatasetBase

CONFIG = {}
CONFIG['url'] = 'http://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz'
CONFIG['out'] = os.path.expanduser('~/.torch/datasets/WISDM_ar_latest.tar.gz')


class _SingleWISDM(data.Dataset):
    path = os.path.expanduser('~/.torch/datasets/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt')
    copy_path = os.path.expanduser('~/.torch/datasets/WISDM_ar_v1.1/WISDM_ar_v1.1_raw_copy.txt')
    all_domain_key = range(36)
    input_shape = (3, 60)
    num_classes = 6

    def __init__(self, domain_key, l_sample, interval):
        assert domain_key in self.all_domain_key
        if not os.path.exists(self.path):
            self.download()
        self.domain_key = domain_key
        self.l_sample = l_sample
        self.interval = interval
        self.prepare_data()

    def download(self):
        output_dir = os.path.dirname(CONFIG['out'])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        wget.download(CONFIG['url'], out=CONFIG['out'])
        tarfile.open(CONFIG['out'], 'r:gz').extractall(output_dir)

    def __getitem__(self, index):
        return self.X[index], self.y[index], self.domain_key

    def __len__(self):
        return len(self.y)

    def prepare_data(self):
        # data cleaning
        if not os.path.exists(self.copy_path):
            shutil.copyfile(self.path, self.copy_path)
            f = open(self.copy_path,"r+")
            d = f.readlines()
            f.seek(0)
            removed = []
            for i in d:
                if ("11,Walking,1867172313000,4.4,4.4," in (i)) or (len(i) < 3):
                    removed.append(i)
                elif ",;" in (i):
                    i = i.replace(',;', '')
                    f.write(i)
                else:
                    i = i.replace(';', '')
                    f.write(i)
            f.truncate()
            f.close()
            # print(removed)
        df = pd.read_csv(self.copy_path, names=['user', 'act', 'time', 'x', 'y', 'z'], header=None)

        # normalization
        for feat in ['x', 'y', 'z']:
            df[feat] = (df[feat] - df[feat].min()) / (df[feat].max() - df[feat].min())

        # group df by labels and domains
        usr_dict = {k:v for v, k in enumerate(df['user'].unique())}
        act_dict = {k:v for v, k in enumerate(df['act'].unique())}
        df_grouped = df.groupby(['user', 'act'])
        usr_act_pairs = df_grouped.groups.keys()

        # collect target domain samples
        df_list = []
        for usr_act_pair in usr_act_pairs:
            if usr_dict[usr_act_pair[0]] == self.domain_key:
                df_list.append(df_grouped.get_group(usr_act_pair))

        x_list = []
        y_list = []
        for df_usr_act in df_list:
            # about l_sample and interval we refer to https://www.sciencedirect.com/science/article/pii/S1568494617305665 
            xs = sampling(df_usr_act[['x', 'y', 'z']], func='sliding', l_sample=self.l_sample, interval=self.interval)
            for x in xs:
                x_list.append(x.values.transpose(1, 0))
                y_list.append(act_dict[df_usr_act['act'].iloc[0]])
        self.X = np.stack(x_list)
        self.y = np.stack(y_list)


class WISDM(DomainDatasetBase):
    """ About: http://www.cis.fordham.edu/wisdm/dataset.php
            - 36 users
            - 6  actions
    """
    SingleDataset = _SingleWISDM

    def __init__(self, domain_keys, require_domain=True, datasets=None, l_sample=60, interval=20):
        self.l_sample = l_sample
        self.interval = interval
        super(WISDM, self).__init__(domain_keys, require_domain, datasets)

    def domain_specific_params(self):
        return {'interval': self.interval,
                'l_sample': self.l_sample}


# copied from: https://github.com/yusuke0519/TL_utils/blob/feature/DPL/pytransfer/datasets/sampling.py
def get_segment(x, start, stop):
    if isinstance(x, pd.DataFrame):
        return x.iloc[start:stop]
    if isinstance(x, np.ndarray):
        return x[start:stop, ...]
    raise Exception("length segment mode requires pd.Dataframe or np.ndarray, not {}.".format(type(x)))


def get_timesegment(x, start, stop):
    if isinstance(x, pd.DataFrame):
        return x.loc[start:stop]
    raise Exception("time segment mode requires pd.Dataframe, not {}.".format(type(x)))


def sliding(x, l_sample, interval):
    start = 0
    stop = start + l_sample
    next_segment = get_segment(x, start, stop)
    while next_segment.shape[0] == l_sample:
        yield next_segment
        start += interval
        stop += interval
        next_segment = get_segment(x, start, stop)


def clip_segment_between(x, start, l_sample):
    for t in start:
        yield get_segment(x, t, t + l_sample)


def clip_time_between(x, start, stop):
    assert len(start) == len(stop), "start and stop must have same length"

    for t1, t2 in zip(start, stop):
        yield get_timesegment(x, t1, t2)


def sampling(x, func, dtype='list', **kwargs):
    if isinstance(x, pd.DataFrame) & ((dtype == 'ndarray') | (dtype == 'np')):
        raise Exception("The combination of x {0} and dtype {1} is not supprted.".format(type(x), dtype))
    if isinstance(func, str):
        func = {
            'sliding': sliding,
            'clips': clip_segment_between,
            'clipt': clip_time_between}.get(func)
        assert func is not None
    X = list(func(x, **kwargs))
    if dtype == 'list':
        return X
    elif (dtype == 'ndarray') | (dtype == 'np'):
        return np.array(X)
    elif (dtype == 'panel') | (dtype == 'pd'):
        index = range(0, len(X))
        return pd.Panel(dict(zip(index, X)))
    else:
        raise Exception("The dtype {} is not supported".format(dtype))
