# # -*- coding: utf-8 -*-
from collections import OrderedDict
import torch
from torch import nn
import torch.utils.data as data

from torch.optim import RMSprop


def get_classifier(layers, input_size, dropout=0.5):
    net = nn.Sequential()
    for i, l in enumerate(layers):
        module = nn.Linear(input_size, l)
        net.add_module('fc{}'.format(i), module)
        input_size = l
        if i != len(layers)-1:
            net.add_module('activation{}'.format(i), nn.ReLU(True))
            net.add_module('dropout{}'.format(i), nn.Dropout(dropout))
    return net


class Discriminator(nn.Module):
    def __init__(self, input_shape, num_output, hiddens=None):
        super(Discriminator, self).__init__()
        if hiddens is None:
            hiddens = []
        if len(input_shape) == 4:
            num_input = input_shape[1] * input_shape[2] * input_shape[3]
        elif len(input_shape) == 2:
            num_input = input_shape[1]
        else:
            raise Exception()
        self.net = get_classifier(hiddens + [num_output], num_input)
        self.activation = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.activation(self.net(x))


def training_step(z, d, D, opt):
    criterion = nn.NLLLoss()
    opt.zero_grad()
    pred_d = D(z)
    loss = criterion(pred_d, d)
    loss.backward()
    opt.step()


def validation_step(z, d, D):
    criterion = nn.NLLLoss()
    pred_d = D(z)
    loss = criterion(pred_d, d).item()
    d_hat = torch.argmax(pred_d, dim=1)
    acc = torch.sum(d == d_hat).item() / (len(d) * 1.0)
    return loss, acc


def check_invariance(E, dataset, num_epoch, val_dataset,
                     hiddens, module_names, num_domains,
                     batch_size=128, lr=0.001, on_gpu=False):
    torch.set_grad_enabled(True)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    for module in module_names:
        getattr(E.feature, module).register_forward_hook(get_activation(module))
    X, y, d = loader.__iter__().__next__()
    if on_gpu:
        X = X.cuda()
        d = d.cuda()
    E(X.float())

    # prepare discriminator
    D = {}
    opt = {}
    for module in module_names:
        D[module] = Discriminator(activation[module].shape, num_domains, hiddens)
        if on_gpu:
            D[module].cuda()
        opt[module] = RMSprop(D[module].parameters(), lr=lr, alpha=0.9)

    # training
    for i in range(num_epoch):
        for X, y, d in loader:
            if on_gpu:
                X = X.cuda()
                d = d.cuda()
            E(X.float())
            for module in module_names:
                z = activation[module].float()
                if on_gpu:
                    z = z.cuda()
                training_step(z, d, D[module], opt[module])

    # validation
    outputs = []
    result = OrderedDict()

    for X, y, d in loader:
        if on_gpu:
            X = X.cuda()
            d = d.cuda()
        E(X.float())
        for module in module_names:
            loss, acc = validation_step(activation[module], d, D[module])
            result['{}-loss-train'.format(module)] = loss
            result['{}-acc-train'.format(module)] = acc
        outputs.append(result)

    for X, y, d in val_loader:
        if on_gpu:
            X = X.cuda()
            d = d.cuda()
        E(X.float())
        for module in module_names:
            loss, acc = validation_step(activation[module], d, D[module])
            result['{}-loss'.format(module)] = loss
            result['{}-acc'.format(module)] = acc
        outputs.append(result)

    avg_result = OrderedDict()
    for k, v in outputs[0].items():
        avg_result[k] = 0.0
    for output in outputs:
        for k, v in output.items():
            avg_result[k] += v
    for k, v in outputs[0].items():
        avg_result[k] /= len(outputs)
    torch.set_grad_enabled(False)
    return avg_result
