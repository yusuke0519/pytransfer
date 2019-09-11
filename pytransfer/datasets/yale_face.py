# # -*- coding: utf-8 -*-

import os
import wget
from glob import glob
from PIL import Image
from zipfile import ZipFile

import numpy as np
import torch.utils.data as data

from base import DomainDatasetBase

CONFIG = {}
CONFIG['url'] = 'http://vision.ucsd.edu/extyaleb/CroppedYaleBZip/CroppedYale.zip'
CONFIG['out'] = os.path.expanduser('~/.torch/datasets')


class _SingleYaleFace(data.Dataset):
    path = os.path.expanduser('~/.torch/datasets/CroppedYale')
    all_domain_condition_dict = {'front': '+000E+00',
                                 'upper right': '+070E+45',
                                 'lower right': '+070E-35',
                                 'upper left': '-070E+45',
                                 'lower left': '-070E-35',
                                 'test': 'test'
                                 }
    all_domain_key = all_domain_condition_dict.keys()
    input_shape = (1, 56, 64)
    num_classes = 38

    def __init__(self, domain_key):
        assert domain_key in self.all_domain_key
        if not os.path.exists(self.path):
            self.download()
        self.domain_key = domain_key
        user_dirs = glob(self.path + '/*')
        images, labels = None, None
        for label, user_dir_name in enumerate(user_dirs):
            user_dir = os.path.join(self.path, user_dir_name)
            user_images = glob(user_dir + '/*')
            for image_path in user_images:
                if (self.domain_key == 'test' and self.is_test_sample(image_path)) or \
                   (self.all_domain_condition_dict[domain_key] in image_path):
                    image = np.array(Image.open(image_path).resize((self.input_shape[1], self.input_shape[2])))
                    image = image.reshape(1, image.shape[0], image.shape[1])
                    # initialize
                    if images is None:
                        images = image
                        labels = np.array([label])
                    else:
                        images = np.concatenate([images, image])
                        labels = np.concatenate([labels, np.array([label])])
                else:
                    pass

        self.X = images
        self.y = labels

    def is_test_sample(self, image_path):
        for domain_key in self.all_domain_key:
            if (domain_key in image_path) or (not 'pgm' in image_path) or ('Ambient' in image_path):
                return False
            else:
                return True

    def download(self):
        wget.download(CONFIG['url'], out=CONFIG['out'])
        zip_file = os.path.join(CONFIG['out'], 'CroppedYale.zip')
        with ZipFile(zip_file) as zf:
            zf.extractall(CONFIG['out'])
        os.remove(zip_file)

    def __getitem__(self, index):
        return self.X[index], self.y[index], self.domain_key

    def __len__(self):
        return len(self.y)


class YaleFace(DomainDatasetBase):
    """ YaleFace Database, used in `VFAE`, `Controllable Invariance` and `Learning unbiased features`.
    Args:
      domain_keys: a list of domains
    """
    SingleDataset = _SingleYaleFace


if __name__ == '__main__':
    dataset = YaleFace(domain_keys=['front', 'upper left', 'upper right', 'lower left', 'lower right'])
    print(dataset[0])
    print(dataset[150])
    dataset = YaleFace(domain_keys=['test'])
    print(dataset[0])
    print(dataset[150])
