# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def generate_points(n, d=2):
    """
    Generate points in R^d.
    """
    return np.random.uniform(-1, 1, (n, 2))


def generate_2dline():
    """
    Generate a line in R^2.
    """
    points = generate_points(2)
    tmp = points[0, :] - points[1, :]
    slope = tmp[1] / tmp[0]
    intercept = points[0, 0] * slope - points[0, 1]
    return slope, intercept


def line_boundary(slope=None, intercept=None):
    """
    Line bound in R^2.
    """
    if slope is None or intercept is None:
        k, b = generate_2dline()

    def boundary(x):
        return x * k + b

    return boundary


class Dataset:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def get_X(self):
        return self.X

    def get_y(self):
        return self.y


class PointsDataset(Dataset):

    def __init__(self, N, d=2, boundary=None):
        """
        Generate a dataset which is labeled by boundary function of size N in R^d.
        """
        assert boundary is not None or d == 2

        X = generate_points(N, d)
        if boundary is None:
            boundary = line_boundary()
        y = np.where(boundary(X[:, 0]) >= X[:, 1], 1, -1)

        self.boundary = boundary
        super(PointsDataset, self).__init__(X, y)

    def get_boundary(self):
        return self.boundary

    def plot(self, ax=None, highlighted=None, hightlight_style=None):
        """
        highlighted: a list of indexes of points which are highlighed.
        hightlight_style: ...
        """
        if ax is None:
            fig, ax = plt.subplots()

        # draw points
        pos = self.X[self.y == 1]
        neg = self.X[self.y == -1]
        ax.scatter(pos[:, 0], pos[:, 1], color='red')
        ax.scatter(neg[:, 0], neg[:, 1], color='blue')
        if highlighted is not None:
            if hightlight_style is None:
                hightlight_style = {
                    'marker': 's',
                    'facecolor': 'none',
                    'edgecolor': 'black',
                    's': 100,
                }
            hl = self.X[highlighted]
            ax.scatter(hl[:, 0], hl[:, 1], **hightlight_style)
        # draw bound
        x = np.linspace(-1.5, 1.5, 20)
        y = self.boundary(x)
        subindex = np.logical_and(y <= 1.5, y >= -1.5)
        ax.plot(x[subindex], y[subindex], color='black', label='target')


def add_bias(X):
    return np.vstack([np.ones(X.shape[0]), X.T]).T


class PLA:
    """
    A perceptron object.
    """
    def __init__(self, Xtransformations=None, ytransformations=None):
        """
        transform_X: a list of transformations applied to X in given order.
        transfrom_y: a list of transformations applied to y in given order.
        """
        self.w = None  # weight
        self.Xtransformations = Xtransformations
        self.ytransformations = ytransformations

    def get_weight(self):
        return self.w

    def transformX(self, X):
        if self.Xtransformations is not None:
            for trans in self.Xtransformations:
                X = trans(X)
        return X

    def transformy(self, y):
        if self.ytransformations is not None:
            for trans in self.ytransformations:
                y = trans(y)
        return y

    def fit(self, X, y, w=None, max_step=200):
        """
        Fit the dataset.
        """
        _X = self.transformX(X)
        _y = self.transformy(y)
        assert _y.shape[0] == _X.shape[0]
        assert w is None or w.shape[0] == _X.shape[1]

        if w is None:
            w = np.zeros(_X.shape[1])
        self.w = w
        count = 0
        misclassified = np.where(self.predict(X) != _y)[0]
        while misclassified.size > 0:
            # randomly pick one misclassified data
            picked = np.random.choice(misclassified)
            self.w += _y[picked] * _X[picked]

            misclassified = np.where(self.predict(X) != _y)[0]
            count += 1
            if count > max_step:
                return False
        return count


    def predict(self, X):
        """
        Predict the label.
        """
        X = self.transformX(X)
        assert self.w is not None
        assert self.w.shape[0] == X.shape[1]

        return np.where(X.dot(self.w) >= 0, 1, -1)

    def error(self, X, y):
        """
        Return the rate of misclassification.
        """
        predicted = self.predict(X)
        y = self.transformy(y)
        return 1 - (y == predicted).sum() / predicted.size

    def plot(self, color='green', label=None, ax=None):
        assert self.w is not None

        if ax is None:
            fig, ax = plt.subplots()
        if label is None:
            label = 'h'

        x = np.linspace(-1.5, 1.5, 20)
        y = (x * self.w[1] + self.w[0]) / (-self.w[2])
        subindex = np.logical_and(y <= 1.5, y >= -1.5)
        ax.plot(x[subindex], y[subindex], color=color, label=label)


# fig, ax = plt.subplots()
# training_set = PointsDataset(10)
# training_set.plot(ax=ax)
# pla = PLA(Xtransformations=(add_bias,))
# pla.fit(training_set.get_X(), training_set.get_y())
# pla.plot(ax=ax)
