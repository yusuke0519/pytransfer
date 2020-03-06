# # -*- coding: utf-8 -*-
from torch import nn


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class Encoder(nn.Module):
    def __init__(self, input_shape, hidden_size=None, activation='relu'):
        super(Encoder, self).__init__()
        if hidden_size is None:
            hidden_size = 1600
        self.hidden_size = hidden_size
        self.input_shape = input_shape
        linear_size = 20 * input_shape[1] * 2

        if activation == 'relu':
            activation = nn.ReLU
        elif activation == 'lrelu':
            activation = nn.LeakyReLU
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(input_shape[0], 50, kernel_size=(1, 5)))
        self.feature.add_module('f_relu1', activation())
        self.feature.add_module('f_pool1', nn.MaxPool2d(kernel_size=(1, 2)))
        self.feature.add_module('f_conv2', nn.Conv2d(50, 40, kernel_size=(1, 5)))
        self.feature.add_module('f_relu2', activation())
        self.feature.add_module('f_pool2', nn.MaxPool2d(kernel_size=(1, 2)))
        self.feature.add_module('f_conv3', nn.Conv2d(40, 20, kernel_size=(1, 3)))
        self.feature.add_module('f_relu3', activation())
        self.feature.add_module('f_drop3', nn.Dropout(0.5))
        self.feature.add_module('flatten', Flatten())
        self.feature.add_module('c_fc1', nn.Linear(linear_size, self.hidden_size))
        self.feature.add_module('c_relu1', activation())
        self.feature.add_module('c_drop', nn.Dropout(0.5))

    def forward(self, input_data):
        feature = self.feature(input_data)
        return feature

    def output_shape(self):
        return (None, self.hidden_size)


class Classifier(nn.Module):
    def __init__(self, num_classes, input_shape):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_shape[1], num_classes),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, input_data):
        return self.classifier(input_data)
