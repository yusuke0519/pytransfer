# # -*- coding: utf-8 -*-
from __future__ import division
from torch import nn


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class Encoder(nn.Module):
    def __init__(self, input_shape, hidden_size=None):
        super(Encoder, self).__init__()

        if hidden_size is None:
            hidden_size = 100
        self.hidden_size = hidden_size

        row = input_shape[2]
        self.input_shape = input_shape
        self.latent_row = ((row - 4) - 4) // 2

        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(input_shape[0], 32, kernel_size=5))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(32, 48, kernel_size=5))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('flatten', Flatten())
        self.feature.add_module('c_fc1', nn.Linear(48*self.latent_row**2, self.hidden_size))
        self.feature.add_module('c_relu1', nn.ReLU(True))

    def forward(self, input_data):
        feature = self.feature(input_data)
        return feature

    def output_shape(self):
        return (None, self.hidden_size)


class Classifier(nn.Module):
    def __init__(self, num_classes, input_shape):
        super(Classifier, self).__init__()
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc2', nn.Linear(input_shape[1], 100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, num_classes))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=-1))

    def forward(self, input_data):
        return self.class_classifier(input_data)


