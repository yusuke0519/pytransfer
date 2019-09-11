# # -*- coding: utf-8 -*-

import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
from sklearn import metrics

from pytransfer.datasets.utils import get_joint_valid_dataloader


class Discriminator(nn.Module):
    def __init__(self, num_domains, input_shape):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_shape[1], 400),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(400, num_domains),
            nn.LogSoftmax()
        )

    def forward(self, input_data):
        return self.discriminator(input_data)


def check_invariance(E, dataset, num_iterations, num_eval, writer=None, test_key=None, current_epoch=None,
                     use_all_data=False):
    E.eval()
    if use_all_data:
        dataset = dataset.__class__(dataset.get('all_domain_key'))
    train_loader, test_loader = get_joint_valid_dataloader(dataset, 0.2, 128)

    D = Discriminator(len(dataset.domain_keys), E.output_shape()).cuda()
    optimizer = optim.RMSprop(D.parameters(), lr=0.001, alpha=0.9)
    criterion = nn.NLLLoss()
    acc_list = []
    cm_list = []

    # ===== train =====
    for i in range(1, num_iterations + 1):
        optimizer.zero_grad()
        X, _, d = train_loader.__iter__().__next__()
        X = Variable(X.float().cuda(), volatile=True)
        z = Variable(E(X).data)
        d = Variable(d.long().cuda())
        d_pred = D(z)
        loss = criterion(d_pred, d)

        loss.backward()
        optimizer.step()

        if i % (num_iterations // 10) == 0:
            # ==== evaluate ====
            ds = []
            pred_ds = []
            for _ in range(num_eval):
                X, _, d = test_loader.__iter__().__next__()
                X = Variable(X.float().cuda())

                pred_d = D(E(X))
                pred_d = np.argmax(pred_d.data.cpu(), axis=1)
                ds.append(d.numpy())
                pred_ds.append(pred_d.numpy())

            d = np.concatenate(ds)
            pred_d = np.concatenate(pred_ds)
            acc = metrics.accuracy_score(d, pred_d)
            cm = metrics.confusion_matrix(d, pred_d)
            acc_list.append(acc)
            cm_list.append(cm)
            print('(check invariance) iter: %.3f, Acc: %.3f' % (i, acc))

    # calculate F-value
    idx = np.argmax(acc_list)
    best_acc = acc_list[idx]
    cm = cm_list[idx]
    cm_pre = cm.astype(np.float32) / cm.sum(axis=0) * 100
    cm_rec = cm.astype(np.float32) / cm.sum(axis=1) * 100
    fvalue = (2 * cm_pre * cm_rec) / (cm_pre + cm_rec)

    print('(check invariance) best Acc: %.3f' % best_acc)
    if (writer is not None) and (test_key is not None) and (current_epoch is not None):
        writer.add_scalar('check_invariance/%s' % test_key, best_acc, current_epoch)
    E.train()
    return {'domain-accuracy': best_acc, 'fvalue': fvalue}


def cross_entropy(q, p):
    """
    :param q: (batch_size, num_class), must be normalized by softmax
    :param p: (batch_size, num_class), must be normalized by softmax
    :return:
    """
    return - torch.mean(torch.sum(q * torch.log(p + 1e-8), 1))


def D_KL(q, p):
    """
    :param q: (batch_size, num_class), must be normalized by softmax
    :param p: (batch_size, num_class), must be normalized by softmax
    :return:
    """
    return - cross_entropy(q, q) + cross_entropy(q, p)
