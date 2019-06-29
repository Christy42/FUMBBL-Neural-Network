import numpy as np


def rand_initial_matrix(size_1, size_2, epsilon):
    return (np.random.rand(size_1, size_2) - 0.5) * epsilon * 2
