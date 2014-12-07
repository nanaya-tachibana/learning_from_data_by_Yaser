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

    def plot(self, ax=None):
        """
        Plot data points and boundary
        """
        if ax is None:
            fig, ax = plt.subplots()

        X, y = self.get_X(), self.get_y()
        # plot points
        plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Paired)
        # plot boundary
        x = np.linspace(-1.5, 1.5, 20)
        y = self.boundary(x)
        subindex = np.logical_and(y <= 1.5, y >= -1.5)
        ax.plot(x[subindex], y[subindex], color='black', linewidth=2, label='target')
