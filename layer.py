import numpy as np

from enum_values import LayerTypes
from forward_prop import sigmoid


# TODO: Add bias (which point exactly? Do I have the write number of nodes)
class Layer:
    def __init__(self, nodes, style, input_no, input_data=False):
        self._style = style
        self._no_nodes = nodes
        self._nodes = input_data if input_data else np.matrix([])
        self._input_no = input_no
        if style != LayerTypes.INPUT:
            self._theta = self._initialise_matrix(0.01)
        else:
            self._theta = np.matrix([[]])

    @property
    def nodes(self):
        return self._nodes

    def _initialise_matrix(self, epsilon):
        return (np.random.rand(self._input_no, self._no_nodes) - 0.5) * epsilon * 2

    def next_step(self, x):
        self._nodes = sigmoid(x * self._theta)

    def update_theta(self):
        pass


class NeuralNet:
    def __init__(self, hidden_layers, input_nodes, output_nodes, hidden_nodes, input_data):
        self._layers = [Layer(input_nodes, LayerTypes.INPUT, 0)] + \
                       [Layer(hidden_nodes, LayerTypes.HIDDEN, input_nodes)] + \
                       [Layer(hidden_nodes, LayerTypes.HIDDEN, hidden_nodes)] * (hidden_layers - 1) + \
                       [Layer(output_nodes, LayerTypes.OUTPUT, hidden_nodes)]
