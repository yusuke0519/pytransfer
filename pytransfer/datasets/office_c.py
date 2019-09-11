# -*- coding: utf-8 -*-

import os
import wget
import shutil
from glob import glob
from PIL import Image

import numpy as np
import rarfile
import torch.utils.data as data

from base import DomainDatasetBase


CONFIG = {}
CONFIG['url'] = 'http://www.stat.ucla.edu/~jxie/iFRAME/code/imageClassification.rar'
CONFIG['base'] = os.path.expanduser('~/.torch/datasets')
CONFIG['rar'] = os.path.join(CONFIG['base'], 'imageClassification.rar')


class _SingleOfficeC(data.Dataset):
    path = os.path.expanduser('~/.torch/datasets/imageClassification/dataset')
    all_domain_key = ['caltech', 'amazon', 'dslr', 'webcam']
    all_class_key = ['back_pack', 'calculator', 'mug', 'keyboard', 'monitor',
                     'mouse', 'headphones', 'laptop_computer', 'bike', 'projector']
    input_shape = (3, 224, 224)
    num_classes = 10

    def __init__(self, domain_key):
        assert domain_key in self.all_domain_key
        if not os.path.exists(self.path):
            self.download_and_preprocess()
        self.domain_key = domain_key

        classes = glob(os.path.join(self.path, domain_key) + '/*')
        classes = filter(lambda x: os.path.basename(x) in self.all_class_key, classes)
        Xs, ys = [], []
        for y, class_path in enumerate(classes):
            images = glob(class_path + '/*.jpg')
            for image_path in images:
                img = Image.open(image_path)

                # convert gray to RGB (Though they are rare)
                if img.mode == 'L':
                    img = img.convert('RGB')

                # resize
                img = self.expand2square(img, (0, 0, 0))
                img = img.resize((self.input_shape[1], self.input_shape[2]))

                X = np.array(img)
                X = np.transpose(X, (2, 0, 1))  # transpose into (num_channels, num_rows, num_cols)
                Xs.append(X)
                ys.append(y)
        self.X = np.stack(Xs)
        self.y = np.stack(ys)

    def __getitem__(self, index):
        x = self.X[index]
        x = x.astype(np.float32) / 255
        y = self.y[index]
        return x, y, self.domain_key

    def __len__(self):
        return len(self.y)

    def download_and_preprocess(self):
        if not os.path.exists(CONFIG['base']):
            os.makedirs(CONFIG['base'])
        wget.download(CONFIG['url'], out=CONFIG['base'])

        rf = rarfile.RarFile(CONFIG['rar'])
        rf.extractall(CONFIG['base'])

        # rename Caltech directories
        current_dir = os.getcwd()
        os.chdir(self.path)
        os.rename('256_ObjectCategories', 'caltech')
        os.chdir('./caltech')
        caltech_dirs = glob('./*')
        for caltech_dir in caltech_dirs:
            if os.path.basename(caltech_dir) == '003.backpack':
                new_c_dir = 'back_pack'
            elif os.path.basename(caltech_dir) == '027.calculator':
                new_c_dir = 'calculator'
            elif os.path.basename(caltech_dir) == '041.coffee-mug':
                new_c_dir = 'mug'
            elif os.path.basename(caltech_dir) == '045.computer-keyboard':
                new_c_dir = 'keyboard'
            elif os.path.basename(caltech_dir) == '046.computer-monitor':
                new_c_dir = 'monitor'
            elif os.path.basename(caltech_dir) == '047.computer-mouse':
                new_c_dir = 'mouse'
            elif os.path.basename(caltech_dir) == '101.head-phones':
                new_c_dir = 'headphones'
            elif os.path.basename(caltech_dir) == '127.laptop-101':
                new_c_dir = 'laptop_computer'
            elif os.path.basename(caltech_dir) == '224.touring-bike':
                new_c_dir = 'bike'
            elif os.path.basename(caltech_dir) == '238.video-projector':
                new_c_dir = 'projector'
            else:
                raise Exception()
            os.rename(caltech_dir, new_c_dir)
        os.chdir(current_dir)

        # move Office datasets
        office_keys = ['amazon', 'dslr', 'webcam']
        for office_key in office_keys:
            office_base = os.path.join(self.path, office_key)
            os.chdir(office_base)
            classes = glob('./images/*')
            for class_dir in classes:
                shutil.move(class_dir, './')
            os.rmdir('./images')
        os.chdir(current_dir)
        return

    @staticmethod
    def expand2square(pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result


class OfficeC(DomainDatasetBase):
    """ Office + Caltech

    Args:
      domain_keys: a list of domains

    """
    SingleDataset = _SingleOfficeC


if __name__ == '__main__':
    dataset = OfficeC(domain_keys=['caltech', 'amazon', 'dslr', 'webcam'])
    from IPython import embed; embed()
