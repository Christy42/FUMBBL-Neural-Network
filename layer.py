import numpy as np


class Layer:
    def __init__(self, nodes, style, input_no):
        self._style = style
        self._nodes = [0] * nodes
        self._input_no = input_no
        if style != "input":
            self._theta = self._initialise_matrix(0.01)
        else:
            self._theta = np.matrix([[]])

    def _initialise_matrix(self, epsilon):
        return (np.random.rand(self._input_no, len(self._nodes)) - 0.5) * epsilon * 2


class NeuralNet:
    def __init__(self, hidden_layers, input_nodes, output_nodes, hidden_nodes):
        self._layers = [Layer(input_nodes, "input", 0)] + [Layer(hidden_nodes, "hidden", input_nodes)] * hidden_layers + \
                       [Layer(output_nodes, "output", hidden_nodes)]
