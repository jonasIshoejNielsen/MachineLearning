import numpy as np
import torch


class MetricLogger:
    """Enables simple result logging and metric calculation for machine learning models."""

    def __init__(self, one_hot=True):
        """Initialize logger with empty confusion matrix.

        :param one_hot: Specifies whether predictions are one-hot encoded or not.
        """
        self.mat = np.zeros((10, 10))
        self.one_hot = one_hot

    def log(self, predicted, target):
        """Log results for provided arguments. Results are added to the confusion matrix, making it possible to do
        incremental logging.

        :param predicted: Model predictions.
        :param target: Ground-truth labels.
        """
        if type(predicted) is torch.Tensor:
            predicted = predicted.detach().numpy()
        if type(target) is torch.Tensor:
            target = target.detach().numpy()

        if self.one_hot:
            predicted = np.argmax(predicted, axis=1)

        for pi, ti in zip(predicted, target):
            self.mat[pi, ti] += 1

    def reset(self):
        """Reset the logger's internal confusion matrix.
        """
        self.mat = np.zeros(self.mat.shape)

    def helperFunc(self, i, precision=True):
        sumDiv=0
        for j in range(len(self.mat)):
            if precision:
                sumDiv += self.mat[j,i]      #mat[y=j][x=i]      //todo check if true
            else:
                sumDiv += self.mat[i,j]      #mat[y=i][x=j]      //todo check if true
        if sumDiv == 0:
            return 0
        return self.mat[i,i]/sumDiv
    
    @property
    def accuracy(self) -> float:
        sumDiv = 0
        for column in self.mat:
            for v in column:
                sumDiv = v
        subDiag = sum(np.diag(self.mat))        
        return subDiag/sumDiv

    @property
    def precision(self) -> np.ndarray:
        return np.array([self.helperFunc(i, True) for i in range(len(self.mat))])

    @property
    def recall(self) -> np.ndarray:
        return np.array([self.helperFunc(i, False) for i in range(len(self.mat))])
            
