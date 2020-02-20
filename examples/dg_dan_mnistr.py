# -*- coding: utf-8 -*-

""" Example for domain generalization using domain adversarial training.

"""
from future.utils import iteritems
import os
from collections import OrderedDict

import numpy as np
from sklearn import metrics
import torch
from torch import nn
from torch.utils import data

import six
import time
import random
import torch
from torch.optim import RMSprop
import torch.utils.data as data
from exp_utils import check_invariance
import logging

import pytorch_lightning as pl
from pytorch_lightning.logging import TensorBoardLogger

from pytransfer.datasets.base import Subset
from pytransfer.datasets import MNISTR
from pytransfer.trainer import Learner
from mnistr_network import Encoder, Classifier
from pytransfer.regularizer.dan import DANReguralizer

def domain_wise_splits(dataset, split_size, random_seed=1234):
    datasets1 = []
    datasets2 = []
    dataset_class = dataset.__class__
    domain_keys = dataset.domain_keys
    # domain_keys = dataset.domain_keys￿
    for domain_key in domain_keys:
        single_dataset = dataset.get_single_dataset(domain_key, **dataset.domain_specific_params())
        len_dataset = len(single_dataset)
        train_size = int(len_dataset * split_size)
        indices = list(range(len_dataset))
        random.shuffle(indices)
        indices = torch.LongTensor(indices)
        # indices2 = indices[train_size:]
        dataset1, dataset2 = Subset(single_dataset, indices[:train_size]), Subset(single_dataset, indices[train_size:])
        datasets1.append(dataset1)
        datasets2.append(dataset2)

    datasets1 = dataset_class(domain_keys=domain_keys, datasets=datasets1, **dataset.domain_specific_params())
    datasets2 = dataset_class(domain_keys=domain_keys, datasets=datasets2, **dataset.domain_specific_params())
    return datasets1, datasets2


def prepare_datasets(train_domain, test_domain):
    train_valid_dataset = MNISTR(train_domain)
    test_dataset = MNISTR(test_domain)

    train_dataset, valid_dataset = domain_wise_splits(train_valid_dataset, 0.8)
    return train_dataset, valid_dataset, test_dataset


class DomainGeneralization(pl.LightningModule):
    def __init__(self, E, M, reguralizers, reguralizers_weight, optim, datasets):
        super(DomainGeneralization, self).__init__()
        self.E = E
        self.M = M
        self.regularizers = OrderedDict()
        for reg, weight in zip (reguralizers, reguralizers_weight):
            self.add_regularizer(reg.__class__.__name__, reg, weight)
        self.criterion = nn.NLLLoss()
        self.optim = optim
        self.datasets_config = datasets
        
        self.datasets = prepare_datasets(datasets['train_domain'], datasets['test_domain'])
        logging.info("Initialized")

    def add_regularizer(self, name, regularizer, alpha):
        assert name not in self.regularizers, "name {} is already registered".format(name)
        self.regularizers[name] = (regularizer, alpha)

    def configure_optimizers(self):
        logging.info("Configure optimizers")
        optim = self.optim
        opt_list = []
        opt_list.append(RMSprop(list(self.E.parameters())+list(self.M.parameters()), lr=optim['lr'], alpha=0.9))

        for reg, _ in self.regularizers.values():
            reg.set_optimizer(RMSprop(filter(lambda p: p.requires_grad, reg.parameters()), lr=optim['lr'], alpha=0.9))
            reg.set_loader(self.datasets[0], self.optim['batch_size'])
            # opt_list.append(RMSprop(filter(lambda p: p.requires_grad, reg.parameters()), lr=optim['lr'], alpha=0.9))
        return opt_list, []

    @pl.data_loader
    def train_dataloader(self):
        dataset = self.datasets[0]
        return data.DataLoader(dataset, batch_size=self.optim['batch_size'], shuffle=True)

    def forward(self, X):
        z = self.E(X)
        y = self.M(z)
        return y

    def training_step(self, batch, batch_idx):
        self.update_regularizers()
        X, y, d = batch
        loss = self.loss(X.float(), y.long(), d.long())
        return {'loss': loss, 'log': {'train_loss': loss}}

    def update_regularizers(self):
        for regularizer, _ in self.regularizers.values():
            regularizer.update()

    def loss(self, X, y, d):
        yhat = self(X)
        y_loss = self.criterion(yhat, y)
        loss = y_loss
        for regularizer, alpha in self.regularizers.values():
            loss += alpha * regularizer.loss(self.E(X), y, d)
        return loss

    def losses(self, X, y, d):
        yhat = self(X)
        y_loss = self.criterion(yhat, y)
        losses = {}
        losses['y'] = y_loss.item()
        for i, (regularizer, alpha) in enumerate(self.regularizers.values()):
            losses[i] = regularizer.loss(self.E(X), y, d).item()
        return losses

    def set_loader(self, dataset, batch_size):
        self.loader = data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True)
        for regularizer, _ in self.regularizers.values():
            if regularizer.loader is None:
                regularizer.set_loader(dataset, batch_size)
        return self.loader

    def get_batch(self, as_variable=True, device='cpu'):
        assert self.loader is not None, "Please set loader before call this function"
        X, y, d = self.loader.__iter__().__next__()
        if as_variable:
            X = X.float().to(device)
            y = y.long().to(device)
            d = d.long().to(device)
        return X, y, d

    def predict_y(self, input_data):
        return self.M(self.E(input_data))

    def evaluate(self, loader, nb_batch=None):
        """
        Evaluate model given data loader

        Parameter
        ---------
        nb_batch : int (default: None)
          # batch to calculate the loss and accuracy
        loader : DataLoader
          data loader

        """
        if nb_batch is None:
            nb_batch = len(loader)
        self.eval()
        result = OrderedDict()

        # evaluate main
        for k, v in iteritems(self._evaluate(loader, nb_batch)):
            result['{}-{}'.format('y', k)] = v

        # evaluate regularizer
        for name, (regularizer, _) in iteritems(self.regularizers):
            for k, v in iteritems(regularizer._evaluate(loader, nb_batch)):
                result['{}-{}'.format(name, k)] = v

        self.train()
        return result

    def _evaluate(self, loader, nb_batch):
        ys = []
        pred_ys = []
        loss = 0
        criterion = nn.NLLLoss()
        with torch.no_grad():
            for i, (X, y, d) in enumerate(loader):
                X = X.float()
                target = y.long()
                pred_y = self.predict_y(X)
                loss += criterion(pred_y, target).item()
                pred_y = np.argmax(pred_y.data, axis=1)
                ys.append(y.numpy())
                pred_ys.append(pred_y.numpy())
                if i+1 == nb_batch:
                    break
            loss /= nb_batch
        # print(nb_batch, loss)

        y = np.concatenate(ys)
        pred_y = np.concatenate(pred_ys)

        result = OrderedDict()
        result['accuracy'] = metrics.accuracy_score(y, pred_y)
        result['f1macro'] = metrics.f1_score(y, pred_y, average='macro')
        result['loss'] = loss
        return result

    def save(self, out, prefix=None):
        names = ['E.pth', 'M.pth']
        if prefix is not None:
            names = [prefix + '-' + x for x in names]
        names = [os.path.join(out, x) for x in names]

        for net, name in zip([self.E, self.M], names):
            torch.save(net.state_dict(), name)
        return names        



if __name__ == '__main__':
    print("Execute example")
    # Parameters
    datasets = {'name': "MNISTR", 'train_domain': ['M0', 'M15', 'M30', 'M45', 'M60'], 'test_domain': ['M75']}
    optim = {'lr': 0.001, 'batch_size': 128, 'num_batch': 100}
    alpha = 1.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    print("Load datasets")

    print("Build model...")
    E = Encoder(MNISTR.get('input_shape'), hidden_size=400)
    M = Classifier(MNISTR.get('num_classes'), E.output_shape())
    M.num_classes = MNISTR.get('num_classes')

    print(E)
    print(M)
    print("Set regularizer")
    discriminator_config = {
        "num_domains": len(datasets['train_domain']),
        "input_shape": E.output_shape(), 'hiddens': [100]}
    reg = DANReguralizer(feature_extractor=E, discriminator_config=discriminator_config)

    model = DomainGeneralization(E, M, [reg], [alpha], optim, datasets)
    logger = TensorBoardLogger("tb_logs", name="my_model")
    trainer = Trainer(logger=logger)

    trainer.fit(model)