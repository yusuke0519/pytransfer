# # -*- coding: utf-8 -*-

import os
import zipfile
import requests
from glob import glob

import numpy as np
from PIL import Image
import torch.utils.data as data

from .base import DomainDatasetBase

CONFIG = {}
CONFIG['url'] = "https://docs.google.com/uc?export=download"
CONFIG['file_id'] = '0B6x7gtvErXgfbF9CSk53UkRxVzg'
CONFIG['out'] = os.path.expanduser('~/.torch/datasets')


def download_file_from_google_drive():
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    def unzip(zip_filename):
        with zipfile.ZipFile(zip_filename) as existing_zip:
            directory = os.path.dirname(zip_filename)
            existing_zip.extractall(directory)

    destination = os.path.join(CONFIG['out'], "pacs.zip")

    session = requests.Session()

    response = session.get(CONFIG['url'], params = { 'id' : CONFIG['file_id'] }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : CONFIG['file_id'], 'confirm' : token }
        response = session.get(CONFIG['url'], params = params, stream = True)

    save_response_content(response, destination)
    unzip(destination)
    os.remove(destination)


class _SinglePACS(data.Dataset):
    path = os.path.expanduser('~/.torch/datasets/kfold')
    all_domain_key = ['art_painting', 'cartoon', 'photo', 'sketch']
    input_shape = (3, 224, 224)
    num_classes = 7

    def __init__(self, domain_key):
        assert domain_key in self.all_domain_key
        if not os.path.exists(self.path):
            download_file_from_google_drive()
        self.domain_key = domain_key
        Xs, ys = [], []

        domain_path = os.path.join(self.path, domain_key)
        classes = glob(domain_path + '/*')
        for y, class_path in enumerate(classes):
            images = glob(class_path + '/*')
            for image_path in images:
                X = np.array(Image.open(image_path))
                X = np.transpose(X, (2, 0, 1))  # transpose into (num_channels, num_rows, num_cols)
                Xs.append(X)
                ys.append(y)
        self.X = np.stack(Xs)
        self.y = np.stack(ys)
        self.augment = False

    def __getitem__(self, index):
        """ Return
                X: shape (num_channels, img_rows, img_cols), normalized into [0, 1]
                y: class_id
                d: domain_key (not id)
        """
        if self.augment:
            x = self.X[index].transpose(1, 2, 0)  # transpose to (row, col, channel)
            x = self.scale_augmentation(x)
            x = self.horizontal_flip(x)
            x = x.transpose(2, 0, 1)
        else:
            x = self.X[index][:, :self.input_shape[1], :self.input_shape[2]]
        x = x.astype(np.float32) / 255

        return x, self.y[index], self.domain_key

    def __len__(self):
        return len(self.y)

    @classmethod
    def horizontal_flip(cls, image, rate=0.5):
        if np.random.rand() < rate:
            image = np.copy(image[:, ::-1, :])
        return image

    @classmethod
    def random_crop(cls, image):
        h, w, _ = image.shape
        top = np.random.randint(0, h - cls.input_shape[1])
        left = np.random.randint(0, w - cls.input_shape[2])

        bottom = top + cls.input_shape[1]
        right = left + cls.input_shape[2]

        image = image[top:bottom, left:right, :]
        return image

    @classmethod
    def scale_augmentation(cls, image, scale_range=(256, 400), crop_size=224):
        scale_size = np.random.randint(*scale_range)
        image = np.array(Image.fromarray(image).resize((scale_size, scale_size)))
        image = cls.random_crop(image, (crop_size, crop_size))
        return image


class PACS(DomainDatasetBase):
    SingleDataset = _SinglePACS

    def __init__(self, domain_keys, require_domain=True, datasets=None):
        super(PACS, self).__init__(domain_keys, require_domain, datasets)

    def use_augmentation(self, flag):
        """
        :param flag: Boolean, when flag set, use data augmentation
        """
        for dataset in self.datasets:
            dataset.augment = flag


class _BiasedSinglePACS(_SinglePACS):
    def __init__(self, domain_key):
        assert domain_key in self.all_domain_key
        if not os.path.exists(self.path):
            download_file_from_google_drive()
        self.domain_key = domain_key
        Xs, ys = [], []

        domain_path = os.path.join(self.path, domain_key)
        classes = glob(domain_path + '/*')

        for y, class_path in enumerate(classes):

            # remove class 2 and 5
            if (domain_key in ['art_painting', 'photo']) and (y == 2 or y == 5):
                continue

            images = glob(class_path + '/*')
            for image_path in images:
                X = np.array(Image.open(image_path))
                X = np.transpose(X, (2, 0, 1))  # transpose into (num_channels, num_rows, num_cols)
                Xs.append(X)
                ys.append(y)
        self.X = np.stack(Xs)
        self.y = np.stack(ys)
        self.augment = False


class BiasedPACS(DomainDatasetBase):
    SingleDataset = _BiasedSinglePACS

    def __init__(self, domain_keys, require_domain=True, datasets=None):
        super(BiasedPACS, self).__init__(domain_keys, require_domain, datasets)

    def use_augmentation(self, flag):
        """
        :param flag: Boolean, when flag set, use data augmentation
        """
        for dataset in self.datasets:
            dataset.augment = flag


if __name__ == '__main__':
    dataset2 = BiasedPACS(BiasedPACS.get('all_domain_key'))
    from IPython import embed; embed()
