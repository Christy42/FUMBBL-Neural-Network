import numpy as np

from procedures.procedure import Procedure
import add_bias


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward_prop(x, theta):
    # TODO: Check that the styles are correct
    x_bias = add_bias.add_bias(x)
    for i in range(len(theta)):
        z = x_bias * theta[i]
        x_bias = sigmoid(z)
