
class Procedure:

    def __init__(self, neural_net):
        self.neural_net = neural_net
        self.neural_net.state.stack.push(self)
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
        for i in range(len(self.neural_net.layer), 1, -1):
            # TODO: Need to add in the next layer
            self.neural_net.stack.push(Sigmoid(self.neural_net, self.neural_net.layer[i]))


class Sigmoid(Procedure):
    def __init__(self, neural_net, layer):
        super().__init__(neural_net)
        self.layer = layer

    def step(self, action):
        self.layer.next_step()


class BackwardProp(Procedure):
    def __init__(self, neural_net):
        super().__init__(neural_net)






