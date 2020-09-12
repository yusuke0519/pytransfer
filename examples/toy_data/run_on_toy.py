# # -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import random
from torch.autograd import Variable
import torch.utils.data as data
import torch
from torch import optim
from datasets import MultipleGaussianSynthesis
from utils import Discriminator
# from utils import makedirs, Encoder, Discriminator
from reguralization import AdversarialCatClassifier, AttributePerceptionLoss, NSAdversarialCatClassifier
from reguralization import SlicedWasserstein, DiscriminativeSlicedWasserstein, PairWiseWasserstein
from torch import nn
from pylab import plt

import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette("Set1")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2})


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Encoder(nn.Module):
    def __init__(self, input_shape, hidden_size=2, num_domains=3):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        for i in range(num_domains):
            self.add_module('dense{}'.format(i), nn.Linear(input_shape[0], hidden_size))

    def forward(self, input_data):
        X, d = input_data
        index = Variable(torch.from_numpy(np.arange(len(d)))).cuda()
        z_list = []
        index_order = []
        for _d in np.unique(d):
            dense = getattr(self, 'dense{}'.format(_d))
            index_order.append(torch.masked_select(index, Variable(d).cuda().long() == _d))
            z_list.append(dense(X[torch.masked_select(index, Variable(d).cuda().long() == _d)]))
        z = torch.cat(z_list)
        _, reindex = torch.sort(torch.cat(index_order))
        return z[reindex]

    def output_shape(self):
        return (None, self.hidden_size)


def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.copy_(torch.eye(2))
        m.bias.data.fill_(0.00)


def plot_scatter(dataset, ax=None, E=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for X, _, d in data.DataLoader(dataset, shuffle=False, batch_size=dataset.num_sample):
        if E is None:
            Z = X
        else:
            Z = E((Variable(X.float().cuda()), d)).data.cpu()
        ax.scatter(Z[:, 0], Z[:, 1])
    return ax


def nll_loss(D, dataset, E=None):
    D.eval()
    if E is not None:
        E.eval()
    criterion = torch.nn.NLLLoss()
    for X, _, d in data.DataLoader(dataset, len(dataset), shuffle=False):
        X = Variable(X).float().cuda()
        d = Variable(d).cuda()
        if E is None:
            pred_d = D(X)
        else:
            pred_d = D(E((X, d.data.cpu())))
        loss = criterion(pred_d, d)
    D.train()
    if E is not None:
        E.train()
    return loss.data.item()


def get_z(dataset, E):
    result = []
    for X, _, d in data.DataLoader(dataset, shuffle=False, batch_size=dataset.num_sample):
        if E is None:
            Z = X
        else:
            Z = E((Variable(X.float().cuda()), d)).data.cpu()
        result.append(Z.numpy())
    return result


def plot_prob_mesh(D, xrange, yrange, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    XX, YY = np.meshgrid(xrange, yrange)
    prob = torch.exp(
        D(Variable(torch.FloatTensor(
            np.concatenate([XX.flatten().reshape(-1, 1), YY.flatten().reshape(-1, 1)], axis=1))).cuda()))
    for i in range(D.num_domains):
        pal = sns.light_palette(sns.color_palette()[i], as_cmap=True)
        Z1 = prob.data.cpu().numpy()[:, i].reshape(XX.shape)
        ax.contour(XX, YY, Z1, cmap=pal)
        # ax.clabel(CS, inline=1, fontsize=10)


if __name__ == '__main__':

    seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('-N', type=int, default=1000)
    parser.add_argument('-K', type=int, default=3)
    parser.add_argument('--kappa', type=int, default=1)
    parser.add_argument('--method', type=str, default="AII")
    parser.add_argument('--sn', action='store_true')
    parser.add_argument('--out', type=str, default='results')
    parser.add_argument('--decay', type=float, default=1.0)

    args = parser.parse_args()    # 4. 引数を解析
    print(args)
    kappa = args.kappa
    num_eval = 100
    num_iterations = 200
    K = args.K
    method = args.method
    sn = args.sn
    num_sample = args.N
    radius = 1.0
    scale = 0.2
    ndim = 2
    hidden_size = 2
    ndim = hidden_size
    discriminator_hiddens = [100]
    decay = args.decay

    folder_name = "{}/toy-{}-{}-{}-balance/{}-{}-{}-{}".format(
        args.out, K, hidden_size, num_sample, method, sn, kappa, decay)
    makedirs(folder_name)
    makedirs(os.path.join(folder_name, 'GVF'))


    # Prepare NN and dataset
    # ## dataset
    dataset = MultipleGaussianSynthesis([i * 360/K for i in range(K)], num_sample, ndim, radius, scale)
    loader = data.DataLoader(dataset, batch_size=128, shuffle=True)

    width = 2
    delta = width / 10.0
    xrange = np.arange(-width, width+delta, delta)
    yrange = np.arange(-width, width+delta, delta)

    # ##NN
    E = Encoder((ndim, ), hidden_size=hidden_size, num_domains=K).cuda()
    E = E.apply(init_weights)

    # Prepare optimization
    optimizer = optim.SGD(E.parameters(), lr=0.1)
    D = Discriminator(num_domains=K, input_shape=E.output_shape(), hiddens=discriminator_hiddens, sn=sn).cuda()

    if method == "AII":
        criterion = AdversarialCatClassifier(D, optimizer=optim.SGD, optimizer_params={'lr':  0.1})

    elif method == "NS":
        criterion = NSAdversarialCatClassifier(D, optimizer=optim.SGD, optimizer_params={'lr':  0.1})

    elif method == "IIDM":
        criterion = AttributePerceptionLoss(D, optimizer=optim.SGD, optimizer_params={'lr':  0.1}, decay=decay)
        full_X, full_y, full_d = data.DataLoader(dataset, batch_size=len(dataset)).__iter__().__next__()
        full_z = E((Variable(full_X).float().cuda(), full_d)).data.cpu()
        centroids = criterion.init_centroids(full_z, full_y.numpy(), full_d.numpy())

    elif method == "SW":
        criterion = SlicedWasserstein()

    elif method == "PWD":
        D = Discriminator(
            num_domains=K, input_shape=E.output_shape(), hiddens=discriminator_hiddens, sn=True, use_softmax=False).cuda()
        criterion = PairWiseWasserstein(D, optimizer=optim.SGD, optimizer_params={'lr':  0.1})

    elif method == "DSW":
        D = Discriminator(
            num_domains=K, input_shape=E.output_shape(), hiddens=discriminator_hiddens, sn=True, use_softmax=False).cuda()
        criterion = DiscriminativeSlicedWasserstein(D, optimizer=optim.SGD, optimizer_params={'lr':  0.1})

    # Initialize D
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), sharex=True, sharey=True)
    for _ in range(num_eval):
        X, _, d = loader.__iter__().__next__()
        X = Variable(X).float().cuda()
        z = E((X, d))
        d = Variable(d).cuda()

        criterion.train(z, d)

    plot_prob_mesh(D, xrange, yrange, ax=ax)
    plot_scatter(dataset, E=E, ax=ax)
    ax.set_title("NLL: {:.3f}".format(nll_loss(D, dataset, E)))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(os.path.join(folder_name, 'initial.pdf'), bbox_inches='tight', pad_inches=0.1)
    plt.close()


    # Train D and E
    nll_list = []
    external_nll_list = []
    z_list = []
    for i in range(num_iterations):
        print("Iteration {}".format(i+1))

        # Update D
        for _ in range(kappa):
            X, _, d = loader.__iter__().__next__()
            X = Variable(X).float().cuda()
            z = E((X, d))
            d = Variable(d).cuda()
            criterion.train(z, d)

        # Update E
        if method == "IIDM":
            optimizer.zero_grad()
            # Update E
            # TODO: change it to update centroids
            if criterion.decay == 1.0:
                full_X, full_y, full_d = data.DataLoader(dataset, batch_size=len(dataset)).__iter__().__next__()
                full_z = E((Variable(full_X).float().cuda(), full_d)).data.cpu()
                centroids = criterion.init_centroids(full_z, full_y.numpy(), full_d.numpy())
            else:
                X, y, d = loader.__iter__().__next__()
                z = E((Variable(full_X).float().cuda(), d))
                centroids = criterion.update_centroids(z, y.numpy(), d.numpy())

            X, y, d = loader.__iter__().__next__()
            X = Variable(X).float().cuda()
            z = E((X, d))
            # d = Variable(d).cuda()
            adv_loss = criterion((z, y), d)
        else:
            optimizer.zero_grad()
            X, _, d = loader.__iter__().__next__()
            X = Variable(X).float().cuda()
            z = E((X, d))
            d = Variable(d).cuda()
            adv_loss = criterion(z, d)

        adv_loss.backward()
        optimizer.step()

        # train external_classifier
        D_eval = Discriminator(num_domains=K, input_shape=E.output_shape(), hiddens=discriminator_hiddens).cuda()
        eval_criterion = AdversarialCatClassifier(D_eval, optimizer=optim.SGD, optimizer_params={'lr':  0.1})
        for _ in range(num_eval):
            X, _, d = loader.__iter__().__next__()
            X = Variable(X).float().cuda()
            z = E((X, d))
            d = Variable(d).cuda()
            eval_criterion.train(z, d)
        external_loss = nll_loss(D_eval, dataset, E)
        print(external_loss)

        nll = nll_loss(D, dataset, E)
        external_nll_list.append(external_loss)
        nll_list.append(nll)
        z_list.append(get_z(dataset, E))

        # plot
        # plot
        if hidden_size == 2:
            fig, ax = plt.subplots(1, 1, figsize=(3, 3), sharex=True, sharey=True)
            plot_prob_mesh(D, xrange, yrange, ax=ax)
            plot_scatter(dataset, E=E, ax=ax)

            # ax.set_title("NLL: {:.3f}({:.3f})".format(nll_loss(D, dataset, E), external_loss))
            ax.set_xlim(-2.0, 2.0)
            ax.set_ylim(-2.0, 2.0)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.savefig(os.path.join(folder_name, 'visualize_{}.pdf'.format(i)), bbox_inches='tight', pad_inches=0.1)


    fig, ax = plt.subplots(1, 1, figsize=(4, 3), sharex=True, sharey=True)

    fontsize = 14
    ax.plot(external_nll_list, 'D-', label='D_eval', lw=3, markevery=10)
    ax.plot(nll_list, '--', label='D', lw=3, markevery=10)
    ax.plot([0, len(nll_list)], [-np.log(1.0/K), -np.log(1.0/K)], color='gray', lw=3, label='Max-ent')
    ax.set_xlabel('#steps', fontsize=fontsize)
    ax.set_ylabel('NLL', fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    ax.set_xlim(0, num_iterations)
    plt.tick_params(labelsize=fontsize)
    fig.savefig(os.path.join(folder_name, 'nll_plot.pdf'), bbox_inches='tight', pad_inches=0.1)
    np.savetxt(os.path.join(folder_name, "D_eval.csv"), np.array(external_nll_list))
    np.savetxt(os.path.join(folder_name, "D.csv"), np.array(nll_list))
