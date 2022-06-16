import os
import pickle
from abc import ABC, abstractmethod
from datetime import datetime

from typing import Callable
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from tqdm import tqdm

from fashionmnist_utils import mnist_reader
from metrics import MetricLogger
from sklearn.metrics import confusion_matrix


class Trainer(ABC):
    """Provides a library-independent API for training and evaluating machine learning classifiers."""

    def __init__(self, model):
        """Creates a new model instance with a unique name and underlying model.

        :param model: Model object to be used in training/prediction/evaluation.
        """
        self.model = model
        self.name = f'{type(model).__name__}-{datetime.now().strftime("%m-%d--%H-%M-%S")}'

    @abstractmethod
    def train(self, *args):
        """Completely trains self.model using internal training data
        """
        ...

    @abstractmethod
    def evaluate(self) -> MetricLogger:
        """Evaluate model on the internal testing data.

        :returns: MetricLogger object with results.
        """
        ...

    @abstractmethod
    def save(self):
        """Save the model object in "models". The filename is given by self.name.
        """
        ...

    @staticmethod
    @abstractmethod
    def load(path: str):
        """Load the model object at the specified file location.

        :param path: Path in "models" directory to load from.
        """
        ...

def printValue (text, value):
    print(text)  
    print(value)  
    print()
    
class SKLearnTrainer(Trainer):
    """Implements the Model API for scikit-learn models."""

    def __init__(self, algorithm):
        super().__init__(algorithm)
        X, y = mnist_reader.load_mnist('data/FashionMNIST/raw', kind='train')       #changed path to include data/

        # Load and split datasets into training, validation, and test set.
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_test, self.y_test = mnist_reader.load_mnist('data/FashionMNIST/raw', kind='t10k')       #changed path to include data/
        
    def train(self):
        xTrain, yTrain = self.X_train, self.y_train
        if False:
            xTrain, yTrain = self.X_train[:1000], self.y_train[:1000]
        #train model
        self.model.fit(xTrain, yTrain)
        #Use model
        prediction = self.model.predict(self.X_val)
        #evaluate model
        metricsLog = MetricLogger(one_hot=False)
        metricsLog.log(prediction, self.y_val)
        printValue("Accuracy",  metricsLog.accuracy)
        printValue("Precision", metricsLog.precision)
        printValue("Recall",    metricsLog.recall)
        #confusionMatrix = confusion_matrix(self.y_val, prediction)

    def evaluate(self):
        #return self.model.score(self.X_test, self.y_test)  #accuracy
        prediction = self.model.predict(self.X_test)
        metricsLog = MetricLogger(one_hot=False)
        metricsLog.log(prediction, self.y_test)
        return metricsLog

    def save(self):
        with open(os.path.join('models', self.name + '.pkl'), 'wb') as file:
            pickle.dump(self.model, file)

    @staticmethod
    def load(path: str) -> Trainer:
        new = SKLearnTrainer(None)
        with open(path, 'rb') as file:
            new.model = pickle.load(file)
            new.name = os.path.basename(path).split('.')[0]
            return new


def get_data(transform, train=True):
    return FashionMNIST(os.getcwd(), train=train, transform=transform, download=True)


class PyTorchTrainer(Trainer):
    """Implements the Model API for PyTorch (torch) models."""

    def __init__(self, nn_module: nn.Module, transform: Callable, optimizer: torch.optim.Optimizer, batch_size: int):
        """Initialize model.

        :param nn_module: torch.nn.Module to use for the model.
        :param transform: torchvision.transforms.Transform to apply to dataset images.
        :param optimizer: torch.optim.Optimizer
        :param batch_size: Batch size to use for datasets.
        """
        super().__init__(nn_module)

        self.transform = transform
        self.batch_size = batch_size
        self.optimizer = optimizer

        # Load datasets
        self.train_data, self.val_data, self.test_data = None, None, None
        self.init_data()

        # Create logger for TensorBoard
        self.logger = SummaryWriter()

    def init_data(self):
        """Method for loading datasets.
        """
        data = get_data(self.transform, True)
        test_data = get_data(self.transform, False)
        val_len = int(len(data) * 0.2)
        
        torch.manual_seed(42)
        train_data, val_data = random_split(data, [len(data) - val_len, val_len])

        self.train_data = DataLoader(train_data, self.batch_size)
        self.val_data = DataLoader(val_data, self.batch_size)
        self.test_data = DataLoader(test_data, self.batch_size)

    def accurracy(self, data):
        """print accurracy like in exercise 11, only for testing"""
        correct = 0.0
        for (x, y) in data:
            out = self.model(x)
            prediction = out.argmax(dim=1)
            correct += (prediction == y).sum().item()
        accuracy = correct/len(get_data(self.transform, False))
        return accuracy
  
    def train(self, epochs: int, debug=True, useEarlyStopping=False):
        metricsLogTraining   = MetricLogger(one_hot=False)
        metricsLogValidation = MetricLogger(one_hot=False)
        old_loss=None
        for e in range(epochs):
            print('epoch', e)
            for i, (x, y) in enumerate(self.train_data):
                out = self.model(x)
                loss = F.cross_entropy(out, y)
                
                #update model
                self.optimizer.zero_grad()  # reset the optimizer's gradient before calculating the gradient.
                loss.backward()
                self.optimizer.step()       # update the model weights.

                prediction = out.argmax(dim=1)
                metricsLogTraining.log(prediction, y)
                if i % 40 == 0 and debug:
                    print(str((i / len(self.train_data))*100)+"% done")
                if i %30 == 0:
                    self.logger.add_scalar("trainer_loss", loss.item() , i+(e*len(self.train_data)))
                    self.logger.add_scalar("trainer_accuracy", self.accurracy(self.test_data) , i+(e*len(self.train_data)))
            
            
            loss = 0
            for i, (x, y) in enumerate(self.train_data):
                out = self.model(x)
                loss += F.cross_entropy(out, y).item()
                prediction = out.argmax(dim=1)
                metricsLogValidation.log(prediction, y)
            
            self.logger.add_scalar("validation_accuracy", self.accurracy(self.test_data)  , e)
            self.logger.add_scalar("validation_loss",     loss  , e)
            metricsLogTraining.reset()
            metricsLogValidation.reset()

            if useEarlyStopping:
                print(loss, old_loss)
                if old_loss is not None and loss>old_loss:
                    return self.logger
                old_loss = loss
            
        return self.logger
        
    def evaluate(self) -> MetricLogger:
        metricsLogTest   = MetricLogger(one_hot=False)
        for i, (x, y) in enumerate(self.test_data):
            out = self.model(x)
            prediction = out.argmax(dim=1)
            metricsLogTest.log(prediction, y)
        return metricsLogTest

    def save(self):
        self.train_data, self.val_data, self.test_data = None, None, None
        self.logger = None

        file_name = os.path.join('models', self.name)
        with open(file_name + '.pkl', 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path: str) -> Trainer:
        with open(path, 'rb') as file:
            new = pickle.load(file)
            new.init_data()
            return new
