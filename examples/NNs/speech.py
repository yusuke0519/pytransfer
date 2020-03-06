# # -*- coding: utf-8 -*-
from torch import nn
from oppG import Flatten


class Encoder(nn.Module):
    def __init__(self, input_shape, hidden_size=None):
        super(Encoder, self).__init__()
        self.input_shape = input_shape
        cnl, row, col, = input_shape
        row_out = (row - 7) // 2 - 3
        col_out = (col - 19) // 2 - 9
        self.out = (None, 64 * (row_out * col_out))
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(cnl, 64, kernel_size=(8, 20)))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_drop1', nn.Dropout(p=0.5))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 64, kernel_size=(4, 10)))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        self.feature.add_module('f_drop1', nn.Dropout(p=0.5))
        self.feature.add_module('f_flat', Flatten())
	if hidden_size is not None:
	    self.feature.add_module('f_fc_drop', nn.Dropout(p=0.5))
	    self.feature.add_module('f_fc', nn.Linear(self.out[1], hidden_size))
	    self.feature.add_module('f_fc_relu', nn.ReLU(inplace=True))
	    self.out = (None, hidden_size)

    def forward(self, input_data):
        feature = self.feature(input_data)
        return feature

    def output_shape(self):
        return self.out


class Classifier(nn.Module):
    def __init__(self, num_classes, input_shape):
        super(Classifier, self).__init__()
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(input_shape[1], num_classes))
        self.class_classifier.add_module('softmax', nn.LogSoftmax(dim=-1))

    def forward(self, feature):
        class_output = self.class_classifier(feature)
        return class_output


