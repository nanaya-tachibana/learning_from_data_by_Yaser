import numpy as np
import matplotlib.pyplot as plt


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

    def plot(self, ax=None):
        assert self.w is not None

        if ax is None:
            fig, ax = plt.subplots()

        # create a mesh to plot in
        h = .02  # step size
        x1_min, x1_max = -3, 3
        x2_min, x2_max = -3, 3
        x1x1, x2x2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                                 np.arange(x2_min, x2_max, h))
        # plot boundary
        labels = self.predict(np.c_[x1x1.ravel(), x2x2.ravel()])
        ax.contourf(x1x1, x2x2, labels.reshape(x1x1.shape),
                    cmap=plt.cm.Paired, alpha=0.8)


#fig, ax = plt.subplots()
#training_set = PointsDataset(10)
#pla = PLA(Xtransformations=(add_bias,))
#pla.fit(training_set.get_X(), training_set.get_y())
#pla.plot(ax=ax)
#training_set.plot(ax=ax)
#ax.legend(loc=0)
