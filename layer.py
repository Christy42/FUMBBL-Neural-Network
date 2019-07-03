import numpy as np

from enum_values import LayerTypes
from forward_prop import sigmoid
from add_bias import add_bias
from stack import Stack


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
        return (np.random.rand(self._input_no+1, self._no_nodes) - 0.5) * epsilon * 2

    def next_step(self, x):
        x_bias = add_bias(x)
        self._nodes = sigmoid(x_bias * self._theta)

    def update_theta(self):
        pass

    def set_theta(self, new_value):
        self._theta = new_value


class NeuralNet:
    def __init__(self, hidden_layers, input_nodes, output_nodes, hidden_nodes, input_data):
        self._layers = [Layer(input_nodes, LayerTypes.INPUT, 0)] + \
                       [Layer(hidden_nodes, LayerTypes.HIDDEN, input_nodes)] + \
                       [Layer(hidden_nodes, LayerTypes.HIDDEN, hidden_nodes)] * (hidden_layers - 1) + \
                       [Layer(output_nodes, LayerTypes.OUTPUT, hidden_nodes)]
        self._input_data = input_data
        self.stack = Stack()

    @property
    def layers(self):
        return self._layers

    def set_layer(self, layer, theta):
        self._layers[layer].set_theta(theta)

    #TODO: This needs to be some sort of class to get pushed
    def sigmoid_layer(self, layer):
        self._layers[layer].next_step(self._layers[layer-1].nodes if i > 0 else self._input_data)
        self.stack.push()

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
            proc.step()
