import numpy as np


class Procedure:

    def __init__(self, neural_net):
        self.neural_net = neural_net
        self.neural_net.stack.push(self)
        self.done = False
        self.initialized = False

    def setup(self):
        pass

    def step(self, action):
        pass


class ForwardProp(Procedure):
    def __init__(self, neural_net):
        super().__init__(neural_net)

    def step(self, action):
        for i in range(len(self.neural_net.layers)-2, 0, -1):
            self.neural_net.stack.push(Sigmoid(self.neural_net, i))


class Sigmoid(Procedure):
    def __init__(self, neural_net, layer):
        super().__init__(neural_net)
        self.layer = layer

    def step(self, action):
        self.neural_net.sigmoid_layer(self.layer)


class Delta(Procedure):
    def __init__(self, neural_net, layer):
        super().__init__(neural_net)
        self.layer = layer

    def step(self, action):
        self.neural_net.back_prop_step(self.layer)


class BackwardProp(Procedure):
    def __init__(self, neural_net):
        super().__init__(neural_net)

    def step(self, action):
        for i in range(2, self.neural_net.size):
            self.neural_net.stack.push(Delta(self.neural_net, i))


class Cost(Procedure):
    def __init__(self, neural_net):
        super().__init__(neural_net)

    def step(self, action):
        # TODO:  Get this cost function as the output from the last layer and the nodes from the second last?
        value = np.multiply(self.neural_net.output_data, np.log(self.neural_net.layers[-2].nodes)) + \
                np.multiply(1-self.neural_net.output_data, np.log(1-self.neural_net.layers[-2].nodes))
        print(-np.sum(value) / np.size(value, 0))
        pass
