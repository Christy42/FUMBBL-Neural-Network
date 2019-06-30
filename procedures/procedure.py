
class Procedure:

    def __init__(self, game):
        self.game = game
        self.game.state.stack.push(self)
        self.done = False
        self.initialized = False

    def setup(self):
        pass

    def step(self, action):
        pass

    def available_actions(self):
        pass
