import numpy as np


def add_bias(X):
    return np.vstack([np.ones(X.shape[0]), X.T]).T


def phi_transform(X):
    """
    Phi transform.(Defined above)
    """
    x1 = X[:, 0]
    x2 = X[:, 1]
    N = x1.size
    return np.vstack([np.ones(N),
                      x1,
                      x2,
                      x1**2,
                      x2**2,
                      x1*x2,
                      np.abs(x1 - x2),
                      np.abs(x1 + x2)]).T


def filter_y(values):
    """
    Select y of which the value is in given values list.
    """
    def f(y):
        subset = False
        for v in values:
            subset = np.logical_or(subset, np.equal(y, v))
        return subset
    return f


def subset(X, y, filter_X=None, filter_y=None):
    """
    Subset data set.
    """
    _X = X
    _y = y
    if filter_X is not None:
        filter = filter_X(_X)
    if filter_y is not None:
        filter = filter_y(_y)
    X = _X[filter]
    y = _y[filter]
    return X, y

# create a subset of number 1 and number 5
#select_1_5 = filter_y((1, 5))
#subset_X, subset_y = subset(X, y, filter_y=select_1_5)
#subset_tX, subset_ty = subset(tX, ty, filter_y=select_1_5) 
