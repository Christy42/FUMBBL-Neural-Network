
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
        for i in range(1, self.neural_net):
            pass


class Sigmoid(Procedure):
    def __init__(self, neural_net):
        super().__init__(neural_net)

    def step(self, action):
        pass


class BackwardProp(Procedure):
    def __init__(self, neural_net):
        super().__init__(neural_net)

