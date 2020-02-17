# -*- coding: utf-8 -*-

import os
import tarfile
import wget

import torch.utils.data as data
from sklearn.feature_extraction.text import HashingVectorizer

from .base import DomainDatasetBase


CONFIG = {}
CONFIG['url'] = 'https://www.cs.jhu.edu/%7Emdredze/datasets/sentiment/unprocessed.tar.gz'
CONFIG['out'] = os.path.expanduser('~/.torch/datasets/amazon')
MAX_VOCAB = 2**18


class _SingleAmazon(data.Dataset):
    path = os.path.expanduser('~/.torch/datasets/amazon/sorted_data')
    all_domain_key = ['books', 'dvd', 'electronics', 'kitchen_&_housewares']
    input_shape = MAX_VOCAB
    num_classes = 2

    def __init__(self, domain_key, vect, vocabulary):
        assert domain_key in self.all_domain_key
        if not os.path.exists(self.path):
            self.download()
        self.domain_key = domain_key
        self.vect = vect

        doc_path = os.path.join(self.path, "%s/processed.review" % self.domain_key)
        f = open(doc_path, 'r')
        self.strings = f.readlines()
        f.close()
        self.X, self.y = self.preprocess(vocabulary)

    def download(self):
        output_dir = CONFIG['out']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        gz_path = os.path.join(CONFIG['out'], "unprocessed.tar.gz")
        if not os.path.exists(gz_path):
            wget.download(CONFIG['url'], out=CONFIG['out'])
        tar = tarfile.open(gz_path, "r:gz")
        tar.extractall(CONFIG['out'])
        tar.close()

    def __getitem__(self, index):
        y = self.y[index]
        x = self.vect.transform([self.X[index]]).toarray().reshape(-1)  # BoW representation
        return x, y, self.domain_key

    def __len__(self):
        return len(self.y)

    def preprocess(self, vocabulary):
        rm_count = 0
        strings = []
        labels = []

        for string in self.strings:
            # get label
            y = string.split(' ')[-1]
            if y == '#label#:negative\n':
                y = 0
            elif y == '#label#:positive\n':
                y = 1
            else:
                raise Exception()

            string = string.split(' ')[:-1]
            # multiply word by its count
            string = [[x[:-2]] * int(x[-1]) for x in string]
            string = [e for inner_list in string for e in inner_list]  # flatten

            # collect items contained in given vocabulary
            string = filter(lambda x: x in vocabulary, string)

            # convert list to string
            if len(string) > 0:
                string = reduce(lambda x, y: x + ' ' + y, string)
            else:
                rm_count += 1
                continue

            strings.append(string)
            labels.append(y)

        if rm_count != 0:
            print('Remove %d samples' % rm_count)
        return strings, labels


class Amazon(DomainDatasetBase):
    """ Amazon review dataset for sentiment classification

    Args:
      domain_keys: a list of domains

    """
    SingleDataset = _SingleAmazon

    def __init__(self, domain_keys, require_domain=True, datasets=None, only_unigram=True, vocabulary=None, vect=None):
        if vect is None:
            self.vect = HashingVectorizer(n_features=MAX_VOCAB)
        else:
            self.vect = vect

        if vocabulary is None:
            self.vocabulary = self.get_vocabulary(domain_keys, only_unigram)
        else:
            self.vocabulary = vocabulary
        assert len(self.vocabulary) < self.vect.n_features
        super(Amazon, self).__init__(domain_keys, require_domain, datasets)

    def domain_specific_params(self):
        return {'vect': self.vect,
                'vocabulary': self.vocabulary}

    def get_vocabulary(self, domain_keys, only_unigram):
        """ get vocabulary which is common in given domain_keys """
        vocabs = []
        for domain_key in domain_keys:

            doc_path = os.path.join(self.SingleDataset.path, "%s/processed.review" % domain_key)
            f = open(doc_path, 'r')
            strings = f.readlines()
            f.close()

            vocab = []
            for string in strings:
                # remove lable
                string = string.split(' ')[:-1]
                # remove count
                string = [x[:-2] for x in string]

                # remove bigram
                if only_unigram:
                    string = filter(lambda x: '_' not in x, string)

                vocab += string

            vocabs.append(set(vocab))

        vocabs = reduce(lambda x, y: x & y, vocabs)
        return vocabs


if __name__ == '__main__':
    train_dataset = Amazon(domain_keys=['books', 'dvd'])
    print(len(train_dataset.vocabulary))

    test_dataset = Amazon(domain_keys=['electronics'], vocabulary=train_dataset.vocabulary, vect=train_dataset.vect)
    print(len(test_dataset.vocabulary))
    from IPython import embed; embed()

    train_dataset2 = Amazon(domain_keys=['books', 'dvd'], only_unigram=False)
    print(len(train_dataset2.vocabulary))
