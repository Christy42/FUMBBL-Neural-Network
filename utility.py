import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def add_bias(x):
    bias = np.matrix([[1] * len(x)]).T
    return np.concatenate((bias, x), axis=1)
