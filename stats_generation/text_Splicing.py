import numpy as np


class Games:
    def __init__(self, text):
        self._games = []
        self._text = text

    def divide_games(self):
        pass


class Game:
    def __init__(self, game_text):
        self._text = game_text
        self._block_for = 0
        self._block_against = 0
        self._cps_for = 0
        self._cps_against = 0
        self._int_for = 0
        self._int_against = 0
        self._cas_for = 0
        self._cas_against = 0
        self._pass_for = 0
        self._pass_against = 0
        self._rush_for = 0
        self._rush_against = 0
        self._turns_for = 0
        self._turn_against = 0
        self._foul_for = 0
        self._foul_against = 0
        self._result = [0, 0, 0]
        self._vector = np.matrix([0] * 16)

    def set_values(self):
        pass
