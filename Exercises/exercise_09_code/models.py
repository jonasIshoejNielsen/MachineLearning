from abc import ABC, abstractmethod

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm


class Model(ABC):

    def __init__(self, criterion):
        self.model = None
        self.coef_ = None
        self.intercept_ = None

        self.criterion = criterion

    @abstractmethod
    def predict(self, x):
        ...

    def fit(self, X, y, epochs=200, batch_size=200):
        input_dim = X.shape[1]
        output_dim = 1

        model = torch.nn.Linear(input_dim, output_dim)

        optim = torch.optim.SGD(model.parameters(), lr=10e-2)

        X = DataLoader(np.float32(X), batch_size=batch_size)
        y = DataLoader(y, batch_size)

        for e in tqdm(range(epochs)):
            for batch_x, batch_y in zip(X, y):
                batch_y = batch_y.view(-1, 1)
                predict = model(batch_x)
                loss = self.criterion(predict, batch_y)
                optim.zero_grad()
                loss.backward()

                optim.step()

        self.model = model
        params = dict(self.model.named_parameters())

        self.intercept_ = params['bias'][0]
        self.coef_ = params['weight'][0]


class LinearModel(Model):

    def __init__(self):
        super().__init__(nn.MSELoss())

    def predict(self, x):
        x = np.reshape(x, (-1, 1))
        out = self.model(torch.tensor(np.float32(x)))
        return out.detach().numpy().reshape(-1)


class LogisticModel(Model):

    def __init__(self):
        super().__init__(nn.BCEWithLogitsLoss())

    def predict(self, x):
        out = self.model(torch.tensor(np.float32(x)))
        out = torch.sigmoid(out)
        return out.detach().numpy().reshape(-1)
