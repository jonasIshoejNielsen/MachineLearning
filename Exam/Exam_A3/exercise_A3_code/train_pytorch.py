from sklearn.linear_model import LogisticRegression
from sklearn.svm import *
from sklearn.neighbors import KNeighborsClassifier
from trainers import *
from tqdm import tqdm

import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import networks
from multiprocessing.dummy import Pool as ThreadPool

import time

def trainModelWithOptimizer(model, optimizer, debug=True, useEarlyStopping=False):
    transform       = transforms.ToTensor()
    batch_size      = 128
    print("1. Made setup")
    trainer         = PyTorchTrainer(model, transform= transform, optimizer=optimizer, batch_size=batch_size)
    print("2. Made trainer")
    trainer.train(6, debug=debug, useEarlyStopping=useEarlyStopping)
    print("3. Trained")
    print("--   Accuracy:   ", trainer.accurracy(trainer.test_data), sep="---")
    print("--   Evaluation: ", trainer.evaluate(), sep="---")
    trainer.save()
    print("4. Saved")
    print("5. Done")

def trainModelSGD(model, momentum=0.0, debug=True, useEarlyStopping=False):
    print("Start")
    learningRate    = 0.01
    optimizer       = optim.SGD(model.parameters(), momentum=momentum, lr=learningRate)
    trainModelWithOptimizer(model, optimizer, debug=debug, useEarlyStopping=useEarlyStopping)


def trainModelRMSProp(model, momentum=None, debug=True, useEarlyStopping=False):
    print("Start")
    learningRate    = 0.01
    optimizer       = optim.RMSprop(model.parameters(),momentum=momentum, lr=learningRate)
    
    trainModelWithOptimizer(model, optimizer, debug=debug, useEarlyStopping=useEarlyStopping)


def task1_5():
    trainModelSGD(Linear())


def task2_2():
    print("start task2_2")
    trainModelSGD(networks.MLPModel())
    print("Done first, start second")
    trainModelSGD(networks.MLPModel(outputNumberHiddenLayer=700))
    print("Done second, start third")
    trainModelSGD(networks.CNNModel())
    print("Done third, start fourth")
    trainModelSGD(networks.CNNModel(outputNumberHiddenLayer1=10, outputNumberHiddenLayer2=14))
    print("Done third, start fourth")
    trainModelSGD(networks.CNNModel(outputNumberHiddenLayerLinear=700))

def task3_3_1_test():
    print("start task3_3_1_test")
    trainModelSGD(networks.CNNModel(), momentum=1.0, debug=False)       
    
    print("Done first, start second")
    trainModelSGD(networks.CNNModel(), momentum=0.75, debug=False)   
    
    print("Done second, start third")
    trainModelSGD(networks.CNNModel(), momentum=0.5, debug=False)
    
    print("Done third, start fourth")
    trainModelSGD(networks.CNNModel(), momentum=0.25, debug=False)
    
    print("Done fourth, start fith")
    trainModelSGD(networks.CNNModel(), momentum=0.0, debug=False)


def task3_3_1():
    print("start task3_3_1")
    trainModelRMSProp(networks.CNNModel(), momentum=1.0, debug=False)            #706.180588722229 seconds
    print("Done first, start second")
    trainModelRMSProp(networks.CNNModel(), momentum=0.75, debug=False)
    print("Done second, start third")
    trainModelRMSProp(networks.CNNModel(), momentum=0.5, debug=False)
    print("Done third, start fourth")
    trainModelRMSProp(networks.CNNModel(), momentum=0.25, debug=False)
    print("Done fourth, start fith")
    trainModelRMSProp(networks.CNNModel(), momentum=0.0, debug=False)

def task3_3_2():
    print("start task3_3_2")
    trainModelSGD(networks.CNNModel3())
    print("Done first, start second")
    trainModelSGD(networks.CNNModel2())
    print("Done second, start third")
    trainModelSGD(networks.CNNModel())

def task3_3_3():
    print("start task3_3_3")
    trainModelSGD(networks.CNNModel_dropout(), useEarlyStopping=True)
    print("Done first, start second")
    trainModelSGD(networks.CNNModel_dropout(p1=0.2), useEarlyStopping=True)
    print("Done second, start third")
    trainModelSGD(networks.CNNModel_dropout(p2=0.2), useEarlyStopping=True)
    print("Done third, start fourth")
    trainModelSGD(networks.CNNModel_dropout(p1=0.8), useEarlyStopping=True)
    
#task1_5()
#task2_2()
#task3_3_1_test()
#task3_3_1()
#task3_3_2()
task3_3_3()


