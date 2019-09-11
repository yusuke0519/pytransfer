# -*- coding: utf-8 -*-

import os
import wget
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import torch.utils.data as data
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, roc_auc_score

from base import DomainDatasetBase

COLUMNS = ["age",
           "workclass",
           "fnlwgt",
           "education",
           "educational-num",
           "marital-status",
           "occupation",
           "relationship",
           "race",
           "gender",
           "capital-gain",
           "capital-loss",
           "hours-per-week",
           "native-country",
           "income"]
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"


class _SingleAdultIncome(data.Dataset):
    path = os.path.expanduser('~/.torch/datasets/adult.data.txt')
    all_domain_dict = {"<30": 0,
                       "30~40": 1,
                       "40~50": 2,
                       "50~60": 3,
                       "60<": 4}
    all_domain_key = all_domain_dict.keys()
    input_shape = 12
    num_classes = 2

    def __init__(self, domain_key):
        assert domain_key in self.all_domain_key
        if not os.path.exists(self.path):
            self.download()
        self.domain_key = domain_key
        self.X, self.y = self.preprocess(self.path, self.all_domain_dict[domain_key])

    def __getitem__(self, index):
        return self.X[index], self.y[index], self.domain_key

    def __len__(self):
        return len(self.y)

    @classmethod
    def download(cls):
        output_dir = os.path.dirname(cls.path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        wget.download(URL, out=cls.path)

    # Reference: https://www.kaggle.com/wenruliu/income-prediction
    @staticmethod
    def preprocess(csv_path, domain_id):
        data = pd.read_csv(csv_path, names=COLUMNS)

        # drop rows including missing values
        for column in COLUMNS:
            if data[column].dtype.name == "object":
                data[column] = data[column].str.replace(" ", "")
        data = data[data["workclass"] != "?"]
        data = data[data["occupation"] != "?"]
        data = data[data["native-country"] != "?"]

        # reduce category within marital-status
        data.replace(['Divorced', 'Married-AF-spouse',
                      'Married-civ-spouse', 'Married-spouse-absent',
                      'Never-married', 'Separated', 'Widowed'],
                     ['not married', 'married', 'married', 'married',
                      'not married', 'not married', 'not married'], inplace=True)

        category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                        'relationship', 'gender', 'native-country', 'income']

        # prepare categorical variables
        for col in category_col:
            b, c = np.unique(data[col], return_inverse=True)
            data[col] = c
        data.loc[data["age"] < 30, "age"] = 0
        data.loc[(data["age"] >= 30) & (data["age"] < 40), "age"] = 1
        data.loc[(data["age"] >= 40) & (data["age"] < 50), "age"] = 2
        data.loc[(data["age"] >= 50) & (data["age"] < 60), "age"] = 3
        data.loc[data["age"] >= 60, "age"] = 4

        # prepare particular domain
        data = data[data["age"] == domain_id]
        predictors = ['workclass', 'education', 'educational-num',
                      'marital-status', 'occupation', 'relationship', 'race', 'gender',
                      'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
        y = data["income"].values
        X = data[predictors].values
        return X, y


class AdultIncome(DomainDatasetBase):
    """ https://archive.ics.uci.edu/ml/datasets/adult
    Args:
      domain_keys: a list of domains
    """
    SingleDataset = _SingleAdultIncome

    @property
    def X(self):
        Xs = []
        for dataset in self.datasets:
            Xs.append(dataset.X)
        return np.concatenate(Xs)

    @property
    def y(self):
        ys = []
        for dataset in self.datasets:
            ys.append(dataset.y)
        return np.concatenate(ys)

    @property
    def d(self):
        ds = []
        for dataset in self.datasets:
            d = self.domain_dict[dataset.domain_key]
            ds += [d] * len(dataset)
        one_hot = np.identity(len(self.datasets))[ds]
        return one_hot


if __name__ == '__main__':
    dataset = AdultIncome(domain_keys=AdultIncome.get_disjoint_domains([]))
    print(dataset[0])
    print(dataset[150])

    # train without d
    X_train = dataset.X
    y_train = dataset.y
    logit_train = sm.Logit(y_train, X_train)
    result_train = logit_train.fit()
    y_train_pred = result_train.predict(X_train)
    y_train_pred = (y_train_pred > 0.5).astype(int)
    acc = accuracy_score(y_train, y_train_pred)
    print("ACC=%f" % (acc))
    auc = roc_auc_score(y_train, y_train_pred)
    print("AUC=%f" % (auc))

    # train with d
    X_train = dataset.X
    d_train = dataset.d
    X_train = np.concatenate([X_train, d_train], axis=1)
    y_train = dataset.y
    logit_train = sm.Logit(y_train, X_train)
    result_train = logit_train.fit()
    y_train_pred = result_train.predict(X_train)
    y_train_pred = (y_train_pred > 0.5).astype(int)
    acc = accuracy_score(y_train, y_train_pred)
    print("ACC=%f" % (acc))
    auc = roc_auc_score(y_train, y_train_pred)
    print("AUC=%f" % (auc))
