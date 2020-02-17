# # -*- coding: utf-8 -*-
import os, zipfile, glob
import wget

import numpy as np
import scipy.io
from sklearn.preprocessing import StandardScaler
import torch.utils.data as data

from .sampling import sampling
from .base import DomainDatasetBase

CONFIG = {}
CONFIG['url'] = 'http://sipi.usc.edu/HAD/USC-HAD.zip'
CONFIG['out'] = os.path.expanduser('~/.torch/datasets/USC')


class _SingleUser(data.Dataset):
    path = CONFIG['out']
    all_domain_key = ['S{}'.format(x) for x in range(1, 15)]
    input_shape = (1, 6, None)

    def __init__(self, domain_key, l_sample, interval):
        assert domain_key in self.all_domain_key

        if not os.path.exists(self.path):
            self.download()
        scaler = StandardScaler()


        self.domain_key = domain_key

        file_list = glob.glob(os.path.join(self.path, self.domain_key.replace('S', 'Subject'), "a*.mat"))
        sensor_readings = [None] * len(file_list)
        activity_labels = [None] * len(file_list)
        for i, f in enumerate(file_list):
            data = scipy.io.loadmat(f)
            sr = data['sensor_readings']
            if scaler is not None:
                sr = scaler.fit_transform(sr)  # Preprocess
            X = sampling(sr, 'sliding', dtype='np', l_sample=l_sample, interval=interval)
	    # X_shape = X.shape
            # X = X.reshape((1, X_shape[0], X_shape[1], X_shape[2])).swapaxes(0, 2)
            y = [int(data['activity_number'][0])-1] * X.shape[0]
            sensor_readings[i] = X
            activity_labels[i] = y

        self.X = np.concatenate(sensor_readings, axis=0)
        self.X = self.X.swapaxes(1, 2)[:, np.newaxis, :, :]
        self.Y = np.concatenate(activity_labels).astype('int')
        self.length = len(self.Y)

    def download(self):
        print("Downloading dataset (It takes a while)")
        wget.download(CONFIG['url'], out=self.path+'.zip', bar=None)
        zf = zipfile.ZipFile(self.path + '.zip', 'r')
        zf.extractall(path=self.path)
        zf.close()

    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.domain_key

    def __len__(self):
        return self.length


class USC(DomainDatasetBase):
    SingleDataset = _SingleUser
    num_classes = 12
    def __init__(self, domain_keys, require_domain=True, datasets=None, l_sample=30, interval=15):
        self.interval = interval
        self.l_sample = l_sample
        super(USC, self).__init__(domain_keys, require_domain, datasets=datasets)
    
    def get_single_dataset(self, domain_key, **kwargs):
        return self.SingleDataset(domain_key=domain_key, **kwargs)

    def domain_specific_params(self):
        return {'l_sample': self.l_sample, 'interval': self.interval}

