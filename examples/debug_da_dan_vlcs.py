# -*- coding: utf-8 -*-

""" Example for domain adaptation using domain adversarial training.


Reference: TODO

"""

import time
import random
import torch
from torch.optim import RMSprop
import torch.utils.data as data
from pytransfer.datasets.base import Subset
from pytransfer.datasets import VLCS
from pytransfer.trainer import DALearner
from vlcs_network import Encoder, Classifier
from pytransfer.reguralizer.dan import DANReguralizer
from exp_utils import da_check_invariance, domain_wise_splits
from tensorboardX import SummaryWriter


def prepare_datasets(source, target, ratio=0.8):
    vlcs_dataset = VLCS([source, target])

    train_dataset, valid_test_dataset = domain_wise_splits(vlcs_dataset, ratio)
    return train_dataset, valid_test_dataset


if __name__ == '__main__':
    print("Execute experiment")
    # Parameters
    source = 'V' # 'V', 'L', 'C', 'S'
    target = 'L'
    optim = {'lr': 0.001, 'batch_size': 64, 'num_batch': 5000}
    alpha = 1.0

    print("Load datasets")
    train_dataset, test_valid_dataset = prepare_datasets(source, target, 0.8)
    train_source_size = train_dataset.cummulative_sizes[0]
    valid_size = test_valid_dataset.cummulative_sizes[0]
    train_target_size = train_dataset.cummulative_sizes[1] - train_source_size
    test_size = test_valid_dataset.cummulative_sizes[1] - valid_size
    data_log = "source (%s) train : %d, valid %d| target (%s) train: %d, test %d" % (source, train_source_size, valid_size, target, train_target_size, test_size)
    print(data_log)
    valid_sampler = data.sampler.SubsetRandomSampler([i for i in range(0, valid_size)]) 
    test_sampler = data.sampler.SubsetRandomSampler([i for i in range(valid_size, test_valid_dataset.cummulative_sizes[1])])
    valid_loader = data.DataLoader(valid_dataset, batch_size=optim['batch_size'], shuffle=True, sampler=valid_sampler)
    test_loader = data.DataLoader(test_dataset, batch_size=optim['batch_size'], shuffle=True, sampler=test_sampler)

    print("Build model...")
    E = Encoder(VLCS.get('input_shape'))
    M = Classifier(VLCS.get('num_classes'), E.output_shape())
    M.num_classes = VLCS.get('num_classes')

    print(E)
    print(M)
    learner = DALearner(E, M).cuda()

    optimizer = RMSprop(learner.parameters(), lr=optim['lr'], alpha=0.9)
    print(optimizer)

    print("Set reguralizer")
    discriminator_config = {
        "num_domains": 2, # Source & Target
        "input_shape": E.output_shape(), 'hiddens': [E.output_shape()[1]]}

    reg = DANReguralizer(learner=learner, discriminator_config=discriminator_config)
    reg_optimizer = RMSprop(filter(lambda p: p.requires_grad, reg.parameters()), lr=optim['lr'], alpha=0.9)
    reg.set_optimizer(reg_optimizer)

    learner.add_reguralizer('d', reg, alpha)
    # log
    writer = SummaryWriter()

    print("Optimization")
    EVALUATE_PER = optim['num_batch'] / 20
    start_time = time.time()

    learner.set_loader(train_dataset, optim['batch_size'])
    for batch_idx in range(optim['num_batch']):
        learner.update_reguralizers()
        optimizer.zero_grad()
        X_s, y_s, X, d = learner.get_batch()
        loss = learner.loss(X_s, y_s, X, d)
        loss.backward()
        optimizer.step()
        learner.losses(X_s, y_s, X, d)
        if (batch_idx+1) % EVALUATE_PER != 0:
            continue
        elapse_train_time = time.time() - start_time
        valid_result = learner.evaluate(valid_loader, None, True)
        test_result = learner.evaluate(test_loader, None, False)
        external_result = da_check_invariance(
            learner.E, train_source_dataset, train_target_dataset, 1000, lr=0.001, hiddens=[E.output_shape()[1]], verbose=0)
        d_log = "valid domain loss: %.4f || external domain acc: %.4f " % (valid_result['d-loss'], external_result['valid-domain-accuracy'])
        # HDivergence

        elapsed_time1 = time.time() - start_time
        elapsed_time2 = time.time() - start_time

        base_log = "%s [%06d, %d s (%d s, %d s)] || valid acc: %.3f, valid loss: %.4f|| test acc: %.3f, test loss: %.4f " % (
            "DAN", batch_idx+1, int(elapsed_time2), int(elapse_train_time), int(elapsed_time1), valid_result['y-accuracy'], valid_result['y-loss'], test_result['y-accuracy'], test_result['y-loss']) 
        print(base_log + ' || ' + d_log)
        writer.add_scalar('c_loss/valid/%s' % source_domain[0], valid_result['y-loss'], batch_idx)
        writer.add_scalar('c_acc/valid/%s' % source_domain[0], valid_result['y-accuracy'], batch_idx)
        writer.add_scalar('c_loss/test/%s' % target_domain[0], test_result['y-loss'], batch_idx)
        writer.add_scalar('c_acc/test/%s' % target_domain[0], test_result['y-accuracy'], batch_idx)
        writer.add_scalar('check_invariance/acc', external_result['valid-domain-accuracy'], batch_idx)

        start_time = time.time()
