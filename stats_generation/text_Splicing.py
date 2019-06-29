import numpy as np


class Games:
    def __init__(self, text):
        self._labels = ["completions", "interceptions", "casualties", "passing", "rushing", "blocks", "fouls", "turns"]

        self._games = []
        self._text = text

    def divide_games(self):
        games = self._text.split("match id")
        for i in range(1, len(games)):
            self._games.append(Game(games[i], self._labels))


class Game:
    def __init__(self, game_text, labels):
        self._text = game_text
        self._labels = labels
        self._values = {}
        self._result = [0, 0, 0]
        self._matrix_row = np.matrix([0] * len(self._labels))
        self._tds = [0, 0]
        self.set_values()
        self.result_matrix()

    @property
    def result(self):
        return self._result

    @property
    def matrix_row(self):
        return self._matrix_row

    def set_values(self):
        sides = self._text.split("performances")
        for i in [1, 3]:
            ext = "_for" if i == 1 else "_against"
            players = sides[i].split("performance")
            for j in range(1, len(players)):
                for label in self._labels:
                    self._values[label + ext] = self._values.get(label + ext, 0) + \
                                                int(players[j].split(label + '="')[1].split('"')[0])
                    self._tds[int((i - 1) / 2)] += int(players[j].split('touchdowns="')[1].split('"')[0])
        self._matrix_row = np.matrix(list(self._values.values()))

    def result_matrix(self):
        self._result = [int(self._tds[0] > self._tds[1]), int(self._tds[0] == self._tds[1]),
                        int(self._tds[0] < self._tds[1])]


f = open("data//games.txt", "r")
text = f.read()
b = Games(text)
b.divide_games()
