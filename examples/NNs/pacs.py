# # -*- coding: utf-8 -*-
from torch import nn
import torch.utils.model_zoo as model_zoo


class Encoder(nn.Module):
    url = 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
    
    def __init__(self, input_shape, hidden_size=None, pretrained=True):
        super(Encoder, self).__init__()
	self.hidden_size = hidden_size
        self.input_shape = input_shape
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )
        self.out = 4096
	if hidden_size is not None:
            self.post = nn.Sequential(
                nn.Dropout(),
                nn.Linear(4096, self.hidden_size),
                nn.ReLU(inplace=True)
            )
            self.out = hidden_size

        if pretrained:
            self.init_weights()

    def init_weights(self):
        weights = model_zoo.load_url(self.url)

        # remove the last fc layer's parameter
        weights.pop("classifier.6.weight")
        weights.pop("classifier.6.bias")

        self.load_state_dict(weights)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        if hasattr(self, 'post'):
            x = self.post(x)
        return x

    def output_shape(self):
        return (None, self.out)


class Classifier(nn.Module):
    url = 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'

    def __init__(self, num_classes, input_shape):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_shape[1], num_classes),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


