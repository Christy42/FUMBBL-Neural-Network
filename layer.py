import numpy as np

from enum_values import LayerTypes
from forward_prop import sigmoid
from add_bias import add_bias
from stack import Stack


class Layer:
    def __init__(self, nodes, style, input_no, input_data=False, output_data=False):
        self._style = style
        self._no_nodes = nodes
        self._nodes = input_data if input_data is not False else np.matrix([])
        self._output_data = output_data if output_data is not False else np.matrix([])
        self._input_no = input_no
        self._error = np.matrix([0] * self._no_nodes)
        self._theta = np.matrix([[]]) if style == LayerTypes.INPUT else self._initialise_matrix(0.01)

    @property
    def nodes(self):
        return self._nodes

    @property
    def error(self):
        return self._error

    def calc_error(self, data):
        if self._style == LayerTypes.OUTPUT:
            print(data)
            print("X")
            print(self.nodes)
            self._error = self._nodes - data
        else:
            print("DAJFKASLFJSDAFKD")
            print(self.theta)
            print(np.transpose(data))
            print("GGGGGGGGGGGGGGGGGGGGGGGGGGG")
            print(np.transpose(data) * self._theta)
            print(np.multiply(self._nodes, 1 - self._nodes))
            self._error = np.multiply(np.transpose(data) * self._theta, np.multiply(self._nodes, 1 - self._nodes))
            print(self._error)

    def update(self):
        self._theta += self._error

    @property
    def theta(self):
        return self._theta

    def _initialise_matrix(self, epsilon):
        return (np.random.rand(self._input_no+1, self._no_nodes) - 0.5) * epsilon * 2

    def next_step(self, x):
        x_bias = add_bias(x)
        self._nodes = sigmoid(x_bias * self._theta)
        print("XXXXX")
        print(self._nodes)

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
        self._output_data = output_data
        self.stack = Stack()

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
        print("BBB")
        print(layer)
        print(self.size)
        print(self._output_data)
        print("AAA")
        self._layers[layer].calc_error(self._output_data if layer == self.size-1 else self._layers[layer+1].error)

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
