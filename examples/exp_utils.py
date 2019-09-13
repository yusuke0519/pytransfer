# # -*- coding: utf-8 -*-
import numpy as np
from sklearn import metrics
from torch import nn, optim
from torch.autograd import Variable
import torch.utils.data as data

from pytransfer.regularizers.utils import Discriminator


def check_invariance(E, dataset, num_iterations, valid_dataset, validation_size=0.5, batch_size=128, lr=0.001, verbose=1, hiddens=None):
    if hiddens is None:
        hiddens = [100]
    E.eval()
    D = Discriminator(len(dataset.domain_keys), E.output_shape(), hiddens=hiddens).cuda()
    train_loader = data.DataLoader(dataset, batch_size=128, shuffle=True)
    test_loader = data.DataLoader(valid_dataset, batch_size=128, shuffle=False)
    optimizer = optim.RMSprop(D.parameters(), lr=lr, alpha=0.9)
    criterion = nn.NLLLoss()
    activation = nn.LogSoftmax(dim=-1)
    # ===== train =====
    for i in range(1, num_iterations+1):
        optimizer.zero_grad()
        X, _, d = train_loader.__iter__().__next__()
        X = Variable(X.float().cuda(), volatile=True)
        z = Variable(E(X).data)
        d = Variable(d.long().cuda())
        d_pred = activation(D(z))
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

