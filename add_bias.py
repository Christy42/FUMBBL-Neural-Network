import numpy as np


def add_bias(x):
    bias = np.matrix([[1] * len(x)]).T
    return np.concatenate((bias, x), axis=1)
