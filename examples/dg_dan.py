# -*- coding: utf-8 -*-

""" Example for domain generalization using domain adversarial training.

"""
from future.utils import iteritems
import os
from collections import OrderedDict
from argparse import ArgumentParser, Namespace

import torch
from torch import nn
from torch.utils import data

import random
from torch.optim import RMSprop
from exp_utils import check_invariance
import logging

import pytorch_lightning as pl

from pytransfer.datasets.base import Subset
from pytransfer import datasets
import NNs
from pytransfer.regularizer.dan import DANReguralizer
from pytransfer.regularizer.mars import EnsembleDAN


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
        random.Random(1234).shuffle(indices)
        indices = torch.LongTensor(indices)
        dataset1, dataset2 = Subset(single_dataset, indices[:train_size]), Subset(single_dataset, indices[train_size:])
        datasets1.append(dataset1)
        datasets2.append(dataset2)

    datasets1 = dataset_class(domain_keys=domain_keys, datasets=datasets1, **dataset.domain_specific_params())
    datasets2 = dataset_class(domain_keys=domain_keys, datasets=datasets2, **dataset.domain_specific_params())
    return datasets1, datasets2


def prepare_datasets(train_domain, test_domain, dataset_cls):
    train_valid_dataset = dataset_cls(train_domain)
    test_dataset = dataset_cls(test_domain)

    train_dataset, valid_dataset = domain_wise_splits(train_valid_dataset, 0.8)
    return train_dataset, valid_dataset, test_dataset


class DomainGeneralization(pl.LightningModule):
    def __init__(self, hparams):
        super(DomainGeneralization, self).__init__()
        Encoder, Classifier = NNs.get(hparams.dataset_name)
        dataset_cls = datasets.get(hparams.dataset_name)
        E = Encoder(dataset_cls.get('input_shape'), hidden_size=400)
        M = Classifier(dataset_cls.get('num_classes'), E.output_shape())
        M.num_classes = dataset_cls.get('num_classes')
        self.E = E
        self.M = M

        if hparams.dataset_name == 'mnistr':
            self.val_layers = ['f_relu1', 'f_relu2', 'c_relu1']
        elif hparams.dataset_name in ['oppG', 'oppL', 'usc']:
            self.val_layers = ['f_relu1', 'f_relu2', 'f_relu3', 'c_relu1']

        self.criterion = nn.NLLLoss()
        self.hparams = hparams
        self.D_hiddens = [int(x) for x in hparams.D_hiddens.split('-')] if hparams.D_hiddens != '' else []
        self.datasets = prepare_datasets(
            dataset_cls.get_disjoint_domains(self.hparams.test_domain), self.hparams.test_domain, dataset_cls)
        self.num_domains = len(dataset_cls.get_disjoint_domains(self.hparams.test_domain))
        self.configure_regularizer()
        logging.info("Initialized")

    def configure_regularizer(self):
        self.regularizers = OrderedDict()
        regs = []
        if self.hparams.reg_name == 'dan':
            discriminator_config = {
                "num_domains": self.num_domains,
                "input_shape": self.E.output_shape(), 'hiddens': self.D_hiddens}
            reg = DANReguralizer(feature_extractor=self.E, discriminator_config=discriminator_config)
            self.add_regularizer('D', reg, self.hparams.reg_weight)
            regs.append(reg)
        elif self.hparams.reg_name == 'dan-ent':
            discriminator_config = {
                "num_domains": self.num_domains,
                "input_shape": self.E.output_shape(), 'hiddens': self.D_hiddens}
            reg = DANReguralizer(feature_extractor=self.E, discriminator_config=discriminator_config, max_ent=True)
            self.add_regularizer('D', reg, self.hparams.reg_weight)
            regs.append(reg)
        elif self.hparams.reg_name == 'ensemble':
            discriminator_config = {
                "num_domains": self.num_domains,
                "input_shape": self.E.output_shape(), 'hiddens': self.D_hiddens}
            reg = EnsembleDAN(
                feature_extractor=self.E, discriminator_config=discriminator_config, num_discriminator=5)
            self.add_regularizer('D', reg, self.hparams.reg_weight)
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
        return [
            data.DataLoader(self.datasets[i+1], batch_size=self.hparams.batch_size, shuffle=False) for i in range(2)]

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
            loss += alpha * regularizer(self.E(X), d)
        return loss

    def training_step(self, batch, batch_idx):
        self.update_regularizers()
        X, y, d = batch
        loss = self.loss(X.float(), y.long(), d.long())
        return {'loss': loss, 'log': {'train_loss': loss}}

    def training_end(self, outputs):
        return (outputs)

    def eval_batch(self, batch, batch_idx):
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

    def eval_end(self, outputs):
        avg_result = OrderedDict()
        if isinstance(outputs[0], list):
            avg_result_list = [self.eval_end(o) for o in outputs]
            for i, result in enumerate(avg_result_list):
                for k, v in result.items():
                    avg_result["{}-{}".format(k, i)] = v
            return avg_result
        for k, v in outputs[0].items():
            avg_result[k] = 0.0
        for output in outputs:
            for k, v in output.items():
                avg_result[k] += v
        for k, v in outputs[0].items():
            avg_result[k] /= len(outputs)
        return avg_result

    def validation_step(self, batch, batch_idx, dataset_idx):
        return self.eval_batch(batch, batch_idx)

    def validation_end(self, outputs):
        avg_result = self.eval_end(outputs)
        for i, D_hiddens in enumerate([[], [400], [800], [400, 100]]):
            logs = check_invariance(
                self.E, self.datasets[0], 10, self.datasets[1], D_hiddens,
                self.val_layers, self.num_domains, on_gpu=self.on_gpu,
                lr=self.hparams.lr
            )
            for k, v in logs.items():
                avg_result['inv{}-{}'.format(i, k)] = v
        return {'log': avg_result}

    def test_step(self, batch, batch_idx):
        return self.eval_batch(batch, batch_idx)

    def test_end(self, outputs):
        avg_result = self.eval_end(outputs)
        self.test_result = avg_result
        return {'log': {'test-' + k: v for k, v in avg_result.items()}, 'test_loss': avg_result['y-loss']}

    @staticmethod
    def add_model_specific_args(parser):
        # dataset
        parser.add_argument('--dataset_name', default='mnistr', type=str)
        parser.add_argument('--test_domain', default='M75', type=str)

        # regularizer
        parser.add_argument('--reg_name', default='dan', type=str)
        parser.add_argument('--reg_weight', default=1.0, type=float)
        parser.add_argument('--D_hiddens', default='400', type=str)
        parser.add_argument('--K', default=1, type=int)

        return parser


def check_finish(hparams, experiment_name):
    # NOTE : This is not a perfect logic. For example, the aborted run is also couted as completed for now.
    logging.info("checking status")
    query = ' and '.join(
        ['params.{}="{}"'.format(k, str(v)) for k, v in vars(hparams).items()])
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return False
    finished_runs = MlflowClient().search_runs(
        experiment_ids=[mlflow.get_experiment_by_name(experiment_name).experiment_id],
        filter_string=query, run_view_type=ViewType.ALL
    )
    logging.info("done")
    return len(finished_runs) > 0


def command_execute(hparams, experiment_name):
    # Check if the task if already finished
    flag = check_finish(hparams, experiment_name)
    if flag and not hparams.F:
        logging.info("The task has been already finished or running")
        logging.info("Skip this run")
        logging.info(hparams)
    else:
        logging.info("Execute this run")
        logging.info(hparams)
        random.seed(hparams.seed)
        torch.manual_seed(hparams.seed)
        torch.cuda.manual_seed(hparams.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Parameters
        print("Build model...")
        model = DomainGeneralization(hparams)
        mlf_logger = MLFlowLogger(experiment_name=EXPERIMENT_NAME)
        # mlf_logger = MLFlowLogger(experiment_name=EXPERIMENT_NAME, tracking_uri="s3://log")
        logging.info("MLF Run ID: {}".format(mlf_logger.run_id))
        if hparams.S is True:
            checkpoint = ModelCheckpoint(
                    filepath=os.path.join(os.getcwd(), EXPERIMENT_NAME, str(mlf_logger.run_id)),
                    save_top_k=-1,
            )
            trainer = pl.Trainer(
                mlf_logger, gpus=1,
                max_epochs=hparams.epoch, early_stop_callback=False, checkpoint_callback=checkpoint)
        else:
            trainer = pl.Trainer(
                mlf_logger, gpus=1,
                max_epochs=hparams.epoch, early_stop_callback=False)

        trainer.fit(model)
        trainer.test()
        print(model.test_result)


def generate_json(hparams, experiment_name):
    import json
    import uuid
    del hparams.__dict__['handler']
    log_dir = hparams.__dict__['dir']
    file_name = "{}.json".format(str(uuid.uuid4()))
    del hparams.__dict__['dir']
    flag = check_finish(hparams, experiment_name)

    print(hparams)
    if flag and not hparams.F:
        logging.info("The task has been already finished or running")
        logging.info("Skip save json file")
        logging.info(hparams)
        return None
    with open(os.path.join(log_dir, file_name), mode="w") as f:
        json.dump(hparams.__dict__, f, indent=4)


def execute_from_json(params, experiment_name):
    import json
    with open(params.f) as f:
        _dict = json.load(f)
        hparams = Namespace(**_dict)
    command_execute(hparams, experiment_name)


if __name__ == '__main__':
    from pytorch_lightning.logging import MLFlowLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
    import mlflow
    from mlflow.tracking.client import MlflowClient
    from mlflow.entities import ViewType

    EXPERIMENT_NAME = 'DG_DAN'
    logging.info("Start")
    parser = ArgumentParser(description="Experimet management")
    subparsers = parser.add_subparsers()

    parser_generate = subparsers.add_parser('gen', help='generate json file from arguments')
    parser_generate.add_argument('--dir', default='./jsons', type=str)
    parser_generate.add_argument('--batch_size', default=128, type=int)
    parser_generate.add_argument('--lr', default=0.001, type=float)
    parser_generate.add_argument('--epoch', default=5, type=int)
    parser_generate.add_argument('--seed', default=1234, type=int)
    parser_generate.add_argument('-F', action='store_true')
    parser_generate.add_argument('-S', action='store_true', help='save model parameter')
    parser_generate = DomainGeneralization.add_model_specific_args(parser_generate)
    parser_generate.set_defaults(handler=generate_json)

    parser_json = subparsers.add_parser('json', help='execute from json config')
    parser_json.add_argument('-f', default='./params.json', type=str)
    parser_json.set_defaults(handler=execute_from_json)

    hparams = parser.parse_args()
    if hasattr(hparams, 'handler'):
        logging.info("Call")
        hparams.handler(hparams, EXPERIMENT_NAME)
    else:
        parser.print_help()
