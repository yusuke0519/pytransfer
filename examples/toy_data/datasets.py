import numpy as np
import torch.utils.data as data


def noisy_dataset(num_sample, ndim, mean, scale=1.0):
    return np.random.normal(mean, scale, size=(num_sample, ndim))


class GaussianSynthesis(data.Dataset):
    num_classes = 1
    input_shape = (None, )
    def __init__(self, domain_key, num_sample, ndim, radius, scale):
        self.num_sample = num_sample
        self.ndim = ndim
        self.radius = radius
        self.domain_key = domain_key
        self.mean = [radius * np.sin(2*np.pi*domain_key/360), radius * np.cos(2*np.pi*domain_key/360)]
        self.X = noisy_dataset(num_sample, ndim, self.mean, scale=scale)
        self._transition = [0, 0]  # TODO: this cause error when ndim!=2
                
    def __getitem__(self, index):
        return self.X[index] - self.transition, 0, self.domain_key
    
    def __len__(self):
        return len(self.X)
    
    def centrize(self, flag=True):
        if flag:
            self.transition = self.mean
        else:
            self.transition = [0, 0]
            
    @property
    def transition(self):
        return self._transition
    
    @transition.setter
    def transition(self, value):
        self._transition = value
    
    
class MultipleGaussianSynthesis(data.ConcatDataset):
    def __init__(self, domain_keys, num_sample=1000, ndim=2, radius=1, scale=0.2):
        self.domain_keys = domain_keys
        self.num_sample = num_sample
        self.ndim = ndim
        self.radius = radius
        self.scale = scale

        # create datasets
        datasets = [None] * len(domain_keys)
        for i, domain_key in enumerate(domain_keys):
            datasets[i] = GaussianSynthesis(domain_key, num_sample, ndim, radius, scale)
        super(MultipleGaussianSynthesis, self).__init__(datasets)
        self.datasets = datasets
        self.domain_dict = dict(zip(self.domain_keys, range(len(self.domain_keys))))
        self.centrized = False

    def __getitem__(self, idx):
        X, y, d = super(MultipleGaussianSynthesis, self).__getitem__(idx)
        
        d = self.domain_dict[d]
        return X, y, d

    def centrize(self, flag=True):
        for dataset in self.datasets:
            dataset.centrize(flag)
        
    @property
    def transition(self):
        return self.datasets[0].transition
    
    @transition.setter
    def transition(self, value):
        for dataset in self.datasets:
            dataset.transition = value
