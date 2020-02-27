# -*- coding: utf-8 -*-

""" Example for domain generalization using domain adversarial training.

"""
from future.utils import iteritems
import os
from collections import OrderedDict
from argparse import ArgumentParser

import torch
from torch import nn
from torch.utils import data

import random
from torch.optim import RMSprop
from exp_utils import check_invariance
import logging

import pytorch_lightning as pl

from pytransfer.datasets.base import Subset
from pytransfer.datasets import MNISTR
from mnistr_network import Encoder, Classifier
from pytransfer.regularizer.dan import DANReguralizer


def domain_wise_splits(dataset, split_size, random_seed=1234):
    datasets1 = []
    datasets2 = []
    dataset_class = dataset.__class__
    domain_keys = dataset.domain_keys

    for domain_key in domain_keys:
        single_dataset = dataset.get_single_dataset(domain_key, **dataset.domain_specific_params())
        len_dataset = len(single_dataset)
        train_size = int(len_dataset * split_size)
        indices = list(range(len_dataset))
        random.shuffle(indices)
        indices = torch.LongTensor(indices)
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
    def __init__(self, E, M, hparams):
        super(DomainGeneralization, self).__init__()
        self.E = E
        self.M = M
        self.criterion = nn.NLLLoss()
        self.hparams = hparams
        self.datasets = prepare_datasets(
            MNISTR.get_disjoint_domains(self.hparams.test_domain), self.hparams.test_domain)
        self.num_domains = len(MNISTR.get_disjoint_domains(self.hparams.test_domain))
        self.configure_regularizer()
        logging.info("Initialized")

    def configure_regularizer(self):
        self.regularizers = OrderedDict()
        regs = []
        if self.hparams.reg_name == 'dan':
            discriminator_config = {
                "num_domains": self.num_domains,
                "input_shape": self.E.output_shape(), 'hiddens': self.hparams.D_hiddens}
            reg = DANReguralizer(feature_extractor=E, discriminator_config=discriminator_config)
            self.add_regularizer(reg.__class__.__name__, reg, self.hparams.reg_weight)
            regs.append(reg)
        self.regs = nn.ModuleList(regs)

    def add_regularizer(self, name, regularizer, alpha):
        assert name not in self.regularizers, "name {} is already registered".format(name)
        self.regularizers[name] = (regularizer, alpha)

    def configure_optimizers(self):
        logging.info("Configure optimizers")
        opt_list = []
        opt_list.append(RMSprop(list(self.E.parameters())+list(self.M.parameters()), lr=self.hparams.lr, alpha=0.9))

        for reg, _ in self.regularizers.values():
            reg.set_optimizer(
                RMSprop(filter(lambda p: p.requires_grad, reg.parameters()), lr=self.hparams.lr, alpha=0.9))
            reg.set_loader(self.datasets[0], self.hparams.batch_size)
        return opt_list, []

    @pl.data_loader
    def train_dataloader(self):
        dataset = self.datasets[0]
        return data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        dataset = self.datasets[1]
        return data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False)

    @pl.data_loader
    def test_dataloader(self):
        dataset = self.datasets[2]
        return data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False)

    def forward(self, X):
        z = self.E(X)
        y = self.M(z)
        return y

    def update_regularizers(self):
        for regularizer, _ in self.regularizers.values():
            regularizer.update(self.on_gpu)

    def loss(self, X, y, d):
        yhat = self(X)
        y_loss = self.criterion(yhat, y)
        loss = y_loss
        for regularizer, alpha in self.regularizers.values():
            loss += alpha * regularizer.loss(self.E(X), y, d)
        return loss

    def training_step(self, batch, batch_idx):
        self.update_regularizers()
        X, y, d = batch
        loss = self.loss(X.float(), y.long(), d.long())
        return {'loss': loss, 'log': {'train_loss': loss}}

    def training_end(self, outputs):
        return (outputs)

    def validation_step(self, batch, batch_idx):
        result = OrderedDict()
        X, y, d = batch
        X = X.float()
        pred_y = self(X)
        criterion = nn.NLLLoss()
        loss = criterion(pred_y, y).item()
        y_hat = torch.argmax(pred_y, dim=1)
        acc = torch.sum(y == y_hat).item() / (len(y) * 1.0)
        result['y-loss'] = loss
        result['y-acc'] = acc

        for name, (reg, _) in iteritems(self.regularizers):
            for k, v in iteritems(reg.validation_step((X, y, d), batch_idx)):
                result['{}-{}'.format(name, k)] = v
        return result

    def validation_end(self, outputs):
        avg_result = OrderedDict()
        for k, v in outputs[0].items():
            avg_result[k] = 0.0
        for output in outputs:
            for k, v in output.items():
                avg_result[k] += v
        for k, v in outputs[0].items():
            avg_result[k] /= len(outputs)

        logs = check_invariance(
            self.E, self.datasets[0], 10, self.datasets[1], self.hparams.D_hiddens,
            ['f_relu1', 'f_relu2', 'c_relu1'], self.num_domains, on_gpu=self.on_gpu,
            lr=self.hparams.lr
        )
        for k, v in logs.items():
            avg_result[k] = v
        return {'log': avg_result, 'val_loss': avg_result['y-loss']}

    def save(self, out, prefix=None):
        names = ['E.pth', 'M.pth']
        if prefix is not None:
            names = [prefix + '-' + x for x in names]
        names = [os.path.join(out, x) for x in names]

        for net, name in zip([self.E, self.M], names):
            torch.save(net.state_dict(), name)
        return names

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        # dataset
        parser.add_argument('--dataset_name', default='mnistr', type=str)
        parser.add_argument('--test_domain', default='M75', type=str)

        # regularizer
        parser.add_argument('--reg_name', default='dan', type=str)
        parser.add_argument('--reg_weight', default=1.0, type=float)
        parser.add_argument('--D_hiddens', default=[400], nargs='*', type=int)

        return parser


if __name__ == '__main__':
    from pytorch_lightning.logging import MLFlowLogger
    from mlflow.tracking.client import MlflowClient
    from mlflow.entities import ViewType

    print("Execute example")
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)

    parser = DomainGeneralization.add_model_specific_args(parser)
    hparams = parser.parse_args()

    print(hparams)
    query = ' and '.join(
        ['params.{}="{}"'.format(k, str(v)) for k, v in vars(hparams).items()] + ['attributes.status="FINISHED"'])
    print(query)
    runs = MlflowClient().search_runs(experiment_ids=["1"], filter_string=query, run_view_type=ViewType.ALL)

    if len(runs) > 0:
        logging.info("The task has been already finished")

    # Parameters
    print("Build model...")
    E = Encoder(MNISTR.get('input_shape'), hidden_size=400)
    M = Classifier(MNISTR.get('num_classes'), E.output_shape())
    M.num_classes = MNISTR.get('num_classes')

    model = DomainGeneralization(E, M, hparams)
    mlf_logger = MLFlowLogger(experiment_name='default')
    trainer = pl.Trainer(mlf_logger, gpus=1, max_epochs=50, early_stop_callback=False, checkpoint_callback=False)

    trainer.fit(model)
