from sklearn.linear_model import LogisticRegression
from sklearn.svm import *
from sklearn.neighbors import KNeighborsClassifier
from trainers import *

def task2_2_test():
    print("Start")
    model = LogisticRegression(max_iter=200)
    print("1. Made setup")
    trainer = SKLearnTrainer(model)
    print("2. Made trainer")
    trainer.train()
    print("3. Trained")
    print("--   Evaluation: ", trainer.evaluate(), sep="---")
    trainer.save()
    print("5. Saved")
    print("6. Done")

def helperFuncTrainModel(model, textToPrint):
    print("Start training and printing: " + textToPrint)
    trainerLogisticRegression = SKLearnTrainer(model)
    trainerLogisticRegression.train()
    trainerLogisticRegression.save()
    print()

def task3_1():
    helperFuncTrainModel(LogisticRegression(max_iter=200), "logistic regression")
    helperFuncTrainModel(LinearSVC(), "Linear kernel")
    helperFuncTrainModel(SVC(kernel="poly"), "Polynomial kernel")
    helperFuncTrainModel(KNeighborsClassifier(), "K nearest neighbors")

#task2_2_test()
task3_1()