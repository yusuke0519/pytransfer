import torch
from torch import nn
from torch.nn import Parameter


def get_classifier(layers, input_size, activation=True, dropout=0.0, sn=False):
    net = nn.Sequential()
    for i, l in enumerate(layers):
        module = nn.Linear(input_size, l)
        if sn:
            module = SpectralNorm(module)
        net.add_module('fc{}'.format(i), module)
        if activation:
            net.add_module('activation{}'.format(i), nn.ReLU(True))
        if dropout != 0.0:
            net.add_module('dropout{}'.format(i), nn.Dropout(dropout))
    return net


class Discriminator(nn.Module):
    def __init__(self, num_domains, input_shape, hiddens, sn=False, dropout=0.0, use_softmax=True, bias=True):
        super(Discriminator, self).__init__()
        self.num_domains = num_domains
        self.discriminator = get_classifier(hiddens, input_shape[1], dropout=dropout, sn=sn)
        module = nn.Linear(hiddens[-1], num_domains, bias=bias)
        if sn:
            module = SpectralNorm(module)
        self.linear = module

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

    def parameters(self):
        return filter(lambda p: p.requires_grad, super(Discriminator, self).parameters())
    

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
