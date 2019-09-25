import numpy as np


class Procedure:
    def __init__(self, neural_net):
        self.neural_net = neural_net
        self.neural_net.stack.push(self)
        self.done = False
        self.initialized = False

    def step(self, action):
        pass


class ForwardProp(Procedure):
    def __init__(self, neural_net):
        super().__init__(neural_net)

    def step(self, action):
        for i in range(len(self.neural_net.layers)-1, 0, -1):
            self.neural_net.stack.push(Sigmoid(self.neural_net, i))


class Sigmoid(Procedure):
    def __init__(self, neural_net, layer):
        super().__init__(neural_net)
        self.layer = layer

    def step(self, action):
        self.neural_net.sigmoid_layer(self.layer)


class BackwardProp(Procedure):
    def __init__(self, neural_net):
        super().__init__(neural_net)

    def step(self, action):
        print("AAA")
        print(self.neural_net.size-1)
        for i in range(self.neural_net.size-1, 0, -1):
            print(i)
            self.neural_net.back_prop_step(i)


class Cost(Procedure):
    def __init__(self, neural_net):
        super().__init__(neural_net)

    def step(self, action):
        # TODO:  Get this cost function as the output from the last layer and the nodes from the second last?
        value = np.multiply(self.neural_net.output_data, np.log(self.neural_net.layers[-2].nodes)) + \
                np.multiply(1-self.neural_net.output_data, np.log(1-self.neural_net.layers[-2].nodes))
        regression = 0
        for i in range(self.neural_net.size-1):
            regression = self.neural_net.lambd / (2 * np.size(value, 0)) * \
                         np.sum(np.square(self.neural_net.layers[i].theta))
        print(-np.sum(value) / np.size(value, 0) + regression)
        print("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")
        print(self.neural_net.layers[-2].theta)


class BackProp(Procedure):
    def __init__(self, neural_net):
        super().__init__(neural_net)

    def step(self, action):
        pass
