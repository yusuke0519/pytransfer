# # -*- coding: utf-8 -*-
import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_shape):
        super(Encoder, self).__init__()
        self.input_dim = input_shape[0]
        self.latent_dim = 2000
        self.fc1 = nn.Sequential(nn.Linear(self.input_dim, self.latent_dim))


    def forward(self, input_data):
        X = input_data.view(-1, self.input_dim)
        X = self.fc1(X)
        return X

    def output_shape(self):
        return (None, self.latent_dim)

class Classifier(nn.Module):
    def __init__(self, num_classes, input_shape, activation='log_softmax'):
        super(Classifier, self).__init__()
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(input_shape[1], input_shape[1]))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_fc2', nn.Linear(input_shape[1], num_classes))
        if activation == 'log_softmax':
            self.class_classifier.add_module('c_log_softmax', nn.LogSoftmax(dim=1))
        elif activation == 'softmax':
            self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))
        elif activation is None:
            pass
        else:
            raise Exception()

    def forward(self, input_data):
        return self.class_classifier(input_data)