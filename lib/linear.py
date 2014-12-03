import numpy as np
from .pla import PLA

class LinearRegression(PLA):
    """
    A linear regression object.
    """
    def __init__(self, lambd=0, Xtransformations=None, ytransformations=None):
        """
        transform_X: a list of transformations applied to X in given order.
        transfrom_y: a list of transformations applied to y in given order.
        """
        self.lambd = lambd  # regularzation coefficient
        self.w = None  # weight
        super().__init__(Xtransformations, ytransformations)

    def fit(self, X, y):
        """
        Fit the dataset.
        """
        X = self.transformX(X)
        y = self.transformy(y)
        assert y.shape[0] == X.shape[0]

        d = X.shape[1]
        self.w = np.linalg.pinv(X.T.dot(X) + self.lambd * np.eye(d)).dot(X.T).dot(y)

    def predict(self, X):
        """
        Predict the value.
        """
        X = self.transformX(X)
        assert self.w is not None
        assert self.w.shape[0] == X.shape[1]

        predicted = X.dot(self.w)
        return predicted

    def error(self, X, y):
        """
        Return RMS error.
        """
        predicted = self.predict(X)
        y = self.transformy(y)
        d = y - predicted
        return np.sqrt(d.dot(d) / d.size)


class LinearClassifier(LinearRegression):
    """
    A binary classifier using linear regression.
    y in {-1, 1}
    """
    def predict(self, X):
        """
        Predict the label of the given dataset.
        """
        predicted = super().predict(X)
        predicted[predicted >= 0] = 1
        predicted[predicted < 0] = -1
        return predicted

    def error(self, X, y):
        """
        Return error rate as the fraction of misclassified data.
        """
        predicted = self.predict(X)
        y = self.transformy(y)
        return 1 - (y == predicted).sum() / predicted.size 