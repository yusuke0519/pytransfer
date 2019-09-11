# # -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch import nn
from torch.utils import data
from torch.autograd import Variable

from pytransfer.learners.utils import get_classifier, SpectralNorm


class Discriminator(nn.Module):
    def __init__(self, num_domains, input_shape, hiddens, sn=False, dropout=0.0, use_softmax=True, label_dim=None, bias=True):
        super(Discriminator, self).__init__()
	self.num_domains = num_domains
        self.discriminator = get_classifier(hiddens, input_shape[1], dropout=dropout, sn=sn)
        module = nn.Linear(hiddens[-1], num_domains, bias=bias)
        if sn:
            module = SpectralNorm(module)
        self.linear = module
        if label_dim is not None:
            self.label_linear = nn.Linear(label_dim, num_domains)
            self.onehot_converter = torch.sparse.torch.eye(label_dim)

        if use_softmax:
            self.activation = nn.LogSoftmax(dim=-1)

    def forward(self, input_data):
        z = self.preactivation(input_data)
        if hasattr(self, 'activation'):
            z = self.activation(z)
        return z

    def preactivation(self, input_data):
        X = input_data
        z = self.discriminator(X)
        z = self.linear(z)
        return z


