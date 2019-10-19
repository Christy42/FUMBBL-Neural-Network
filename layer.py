import numpy as np

from enum_values import LayerTypes
from utility import sigmoid, add_bias
from stack import Stack


class Layer:
    def __init__(self, nodes, style, input_no, input_data=False, output_data=False):
        self._style = style
        self._no_nodes = nodes
        self._nodes = input_data if input_data is not False else np.matrix([])
        self._output_data = output_data if output_data is not False else np.matrix([])
        self._input_no = input_no
        self._input_bias = np.matrix([[]])
        self._error = np.matrix([0] * self._no_nodes)
        self._delta = self._error
        self._z_val = False
        self._theta = np.matrix([[]]) if style == LayerTypes.INPUT else self._initialise_matrix(0.01)

    @property
    def nodes(self):
        return self._nodes

    @property
    def input_bias(self):
        return self._input_bias

    @property
    def error(self):
        return self._error

    def calc_error(self, data, follow_theta=False):
        if self._style == LayerTypes.OUTPUT:
            self._delta = self._nodes - data
        else:
            self._delta = np.multiply(data * np.transpose(follow_theta)[:, 1:],
                                      np.multiply(sigmoid(self._z_val), 1 - sigmoid(self._z_val)))

    def update(self):
        self._theta += self._error

    def calc_error_term(self, prev_nodes, lambd):
        self._error = np.transpose(self._delta) * prev_nodes / np.size(self._nodes, 0)

        theta_less_bias = np.concatenate((np.zeros(np.size(np.transpose(self._theta)[:, 1:], 0))[:, np.newaxis], np.transpose(self._theta)[:, 1:]), axis=1)

        self._error += lambd * theta_less_bias / np.size(self._nodes, 0)

    @property
    def theta(self):
        return self._theta

    @property
    def delta(self):
        return self._delta

    def amend_theta(self, i, j, epsilon):
        self._theta[i, j] += epsilon

    def _initialise_matrix(self, epsilon):
        return (np.random.rand(self._input_no+1, self._no_nodes) - 0.5) * epsilon * 2

    def next_step(self, x):
        self._input_bias = add_bias(x)
        self._z_val = self.input_bias * self._theta
        self._nodes = sigmoid(self._z_val)

    @property
    def theta_less_bias(self):
        return self._theta[1:, :]

    def update_theta(self):
        pass

    def prev_step(self):
        pass

    def set_theta(self, new_value):
        self._theta = new_value


class NeuralNet:
    def __init__(self, hidden_layers, input_nodes, output_nodes, hidden_nodes, input_data, output_data):
        self._layers = [Layer(input_nodes, LayerTypes.INPUT, 0, input_data=input_data)] + \
                       [Layer(hidden_nodes, LayerTypes.HIDDEN, hidden_nodes)] * (hidden_layers - 1) + \
                       [Layer(output_nodes, LayerTypes.OUTPUT, hidden_nodes, output_data=output_data)]
        self._input_data = input_data
        self._lambda = 1
        self.cost = 1000
        self._output_data = output_data
        self.stack = Stack()

    def calculate_deltas(self, layer):
        self.layers[layer].calc_error_term(add_bias(self.layers[layer-1].nodes), self._lambda)

    @property
    def layers(self):
        return self._layers

    @property
    def lambd(self):
        return self._lambda

    @property
    def output_data(self):
        return self._output_data

    def set_layer(self, layer, theta):
        self._layers[layer].set_theta(theta)

    # TODO: This needs to be some sort of class to get pushed
    def sigmoid_layer(self, layer):
        self._layers[layer].next_step(self._layers[layer-1].nodes if layer > 0 else self._input_data)

    def back_prop_step(self, layer):
        self._layers[layer].calc_error(self._output_data if layer == self.size-1 else self._layers[layer+1].delta,
                                       False if layer == self.size - 1 else self.layers[layer+1].theta)
        self.layers[layer].update()

    def amend_theta(self, layer, i, j, epsilon):
        self._layers[layer].amend_theta(i, j, epsilon)

    @property
    def size(self):
        return len(self.layers)

    # TODO: Actually do this part
    def step(self):
        """
        Runs the match by taking from the stack
        :return:
        """

        while True:
            # Check if the process is over
            if self.stack.is_empty:
                # TODO: Probably should return something at the end
                return 1

            # Check the next item on the stack and run it.
            proc = self.stack.peek
            # Do action
            self.stack.pop()
            proc.step(1)
