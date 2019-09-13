# -*- coding: utf-8 -*-

""" Example for domain generalization using domain adversarial training.


Reference: TODO

"""

import time
import random
import torch
from torch.optim import RMSprop
import torch.utils.data as data
from pytransfer.datasets.base import Subset
from pytransfer.datasets import MNISTR
from pytransfer.trainer import Learner
from mnistr_network import Encoder, Classifier
from pytransfer.reguralizer.dm import IIDMPlus
from exp_utils import check_invariance


def domain_wise_splits(dataset, split_size, random_seed=1234):
    datasets1 = []
    datasets2 = []
    dataset_class = dataset.__class__
    domain_keys = dataset.domain_keys

    for domain_key in domain_keys:
        single_dataset = dataset.get_single_dataset(domain_key, **dataset.domain_specific_params())
        len_dataset = len(single_dataset)
        train_size = int(len_dataset * split_size)
        indices = range(len_dataset)
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


if __name__ == '__main__':
    print("Execute example")
    # Parameters
    train_domain = ['M0', 'M15', 'M30', 'M45', 'M60']
    test_domain = ['M75']
    optim = {'lr': 0.001, 'batch_size': 128, 'num_batch': 5000}
    alpha = 1.0

    print("Load datasets")
    train_dataset, valid_dataset, test_dataset = prepare_datasets(train_domain, test_domain)
    print(len(train_dataset), len(valid_dataset), len(test_dataset))
    valid_loader = data.DataLoader(valid_dataset, batch_size=optim['batch_size'], shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=optim['batch_size'], shuffle=True)

    print("Build model...")
    E = Encoder(MNISTR.get('input_shape'), hidden_size=400)
    M = Classifier(MNISTR.get('num_classes'), E.output_shape())
    M.num_classes = MNISTR.get('num_classes')

    print(E)
    print(M)
    learner = Learner(E, M).cuda()

    optimizer = RMSprop(learner.parameters(), lr=optim['lr'], alpha=0.9)
    print(optimizer)

    print("Set reguralizer")
    discriminator_config = {
        "num_domains": len(train_dataset.datasets),
        "input_shape": E.output_shape(), 'hiddens': [100]}

    reg = IIDMPlus(learner=learner, discriminator_config=discriminator_config)
    reg_optimizer = RMSprop(filter(lambda p: p.requires_grad, reg.parameters()), lr=optim['lr'], alpha=0.9)
    reg.set_optimizer(reg_optimizer)

    learner.add_reguralizer('d', reg, alpha)

    print("Optimization")
    EVALUATE_PER = optim['num_batch'] / 20
    start_time = time.time()

    learner.set_loader(train_dataset, optim['batch_size'])
    for batch_idx in range(optim['num_batch']):
        learner.update_reguralizers()
        optimizer.zero_grad()
        X, y, d = learner.get_batch()
        loss = learner.loss(X, y, d)
        loss.backward()
        optimizer.step()
        learner.losses(X, y, d)
        if (batch_idx+1) % EVALUATE_PER != 0:
            continue
        # train_result = evaluate(learner, train_loader, 100)
        elapse_train_time = time.time() - start_time
        valid_result = learner.evaluate(valid_loader, None)
        test_result = learner.evaluate(test_loader, None)
        external_result = check_invariance(
            learner.E, train_dataset, 1000, valid_dataset=valid_dataset, lr=0.001, hiddens=[800], verbose=0)
        d_log = "domain d: %.4f || external: %.4f " % (valid_result['d-loss'], external_result['valid-domain-accuracy'])

        # HDivergence
        elapsed_time1 = time.time() - start_time
        elapsed_time2 = time.time() - start_time

        base_log = "%s [%06d, %d s (%d s, %d s)] || valid acc: %.3f, valid loss: %.4f|| test acc: %.3f, test loss: %.4f " % (
            "DAN", batch_idx+1, int(elapsed_time2), int(elapse_train_time), int(elapsed_time1), valid_result['y-accuracy'], valid_result['y-loss'], test_result['y-accuracy'], test_result['y-loss']) 
        print(base_log + ' || ' + d_log)

        start_time = time.time()
