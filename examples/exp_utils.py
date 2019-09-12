# # -*- coding: utf-8 -*-
import random
import numpy as np
from sklearn import metrics
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.utils.data as data

from pytransfer.datasets.base import Subset
from pytransfer.reguralizer.utils import Discriminator


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
        dataset1, dataset2 = Subset(single_dataset, indices[:train_size]), Subset(single_dataset, indices[train_size:])
        datasets1.append(dataset1)
        datasets2.append(dataset2)

    datasets1 = dataset_class(domain_keys=domain_keys, datasets=datasets1, **dataset.domain_specific_params())
    datasets2 = dataset_class(domain_keys=domain_keys, datasets=datasets2, **dataset.domain_specific_params())
    return datasets1, datasets2

def check_invariance(E, dataset, num_iterations, valid_dataset, validation_size=0.5, batch_size=128, lr=0.001, verbose=1, hiddens=None):
    if hiddens is None:
        hiddens = [100]
    E.eval()
    D = Discriminator(len(dataset.domain_keys), E.output_shape(), hiddens=hiddens).cuda()
    train_loader = data.DataLoader(dataset, batch_size=128, shuffle=True)
    test_loader = data.DataLoader(valid_dataset, batch_size=128, shuffle=False)
    optimizer = optim.RMSprop(D.parameters(), lr=lr, alpha=0.9)
    criterion = nn.NLLLoss()
    # ===== train ===== 
    for i in range(1, num_iterations+1):
        optimizer.zero_grad()
        X, _, d = train_loader.__iter__().__next__()
        X = Variable(X.float().cuda(), volatile=True)
        z = Variable(E(X).data)
        d = Variable(d.long().cuda())
        d_pred = D(z)
        loss = criterion(d_pred, d)

        loss.backward()
        optimizer.step()

        if i % (num_iterations//10) == 0:
            d_pred = np.argmax(d_pred.data.cpu().numpy(), axis=1)
            acc = metrics.accuracy_score(d.data.cpu().numpy(), d_pred)
            if verbose > 0:
                print('(check invariance) iter: %.3f, Acc: %.3f' % (i, acc))

    # ==== evaluate ====
    result = {}

    # ==== evaluate with validation====
    ds = []
    pred_ds = []
    for X, _, d in test_loader:
        X = Variable(X.float().cuda())
    
        pred_d = D(E(X))
        pred_d = np.argmax(pred_d.data.cpu(), axis=1)
        ds.append(d.numpy())
        pred_ds.append(pred_d.numpy())
        
    d = np.concatenate(ds)
    pred_d = np.concatenate(pred_ds)
    acc = metrics.accuracy_score(d, pred_d)
    E.train()
    result['valid-domain-accuracy'] = acc
    if verbose > 0:
        print('(check invariance) Test Acc: %.3f' % (acc))
    return result

def da_check_invariance(E, source, target, num_iterations, validation_size=0.5, batch_size=128, lr=0.001, verbose=1, hiddens=None):
    if hiddens is None:
        hiddens = [100]
    E.eval()
    D = Discriminator(2, E.output_shape(), hiddens=hiddens).cuda() # domain adaptation
    # dataset
    train_source_dataset, valid_source_dataset = domain_wise_splits(source, validation_size)
    train_target_dataset, valid_train_dataset = domain_wise_splits(target, validation_size)
    # dataloader
    train_source_loader = data.DataLoader(train_source_dataset, batch_size=64, shuffle=True)
    train_target_loader = data.DataLoader(train_target_dataset, batch_size=64, shuffle=True)
    test_source_loader = data.DataLoader(valid_source_dataset, batch_size=64, shuffle=False)
    test_target_loader = data.DataLoader(valid_train_dataset, batch_size=64, shuffle=False)
    optimizer = optim.RMSprop(D.parameters(), lr=lr, alpha=0.9)
    criterion = nn.NLLLoss()
    # ===== train ===== 
    for i in range(1, num_iterations+1):
        optimizer.zero_grad()
        X_s, _, d_s = train_source_loader.__iter__().__next__()
        X_t, _, d_t = train_target_loader.__iter__().__next__()
        X = Variable(torch.cat((X_s, X_t),0).float().cuda(), volatile=True)
        z = Variable(E(X).data)
        d = Variable(torch.cat((d_s, (d_t + 1)), 0).long().cuda())# target label
        d_pred = D(z)
        loss = criterion(d_pred, d)

        loss.backward()
        optimizer.step()

        if i % (num_iterations//10) == 0:
            d_pred = np.argmax(d_pred.data.cpu().numpy(), axis=1)
            acc = metrics.accuracy_score(d.data.cpu().numpy(), d_pred)
            if verbose > 0:
                print('(check invariance) iter: %.3f, Acc: %.3f' % (i, acc))

    # ==== evaluate ====
    result = {}

    # ==== evaluate with validation====
    ds = []
    pred_ds = []
    for X, _, d in test_source_loader:
        X = Variable(X.float().cuda())
        pred_d = D(E(X))
        pred_d = np.argmax(pred_d.data.cpu(), axis=1)
        ds.append(d.numpy())
        pred_ds.append(pred_d.numpy())
    
    for X, _, d in test_target_loader:
        X = Variable(X.float().cuda())
        pred_d = D(E(X))
        pred_d = np.argmax(pred_d.data.cpu(), axis=1)
        ds.append((d+1).numpy())
        pred_ds.append(pred_d.numpy())

    d = np.concatenate(ds)
    pred_d = np.concatenate(pred_ds)
    acc = metrics.accuracy_score(d, pred_d)
    E.train()
    result['valid-domain-accuracy'] = acc
    if verbose > 0:
        print('(check invariance) Test Acc: %.3f' % (acc))
    return result