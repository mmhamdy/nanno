import numpy as np

class ANN:
    def __init__(self, layers=None, expected_range=(-1, 1)):
        self.layers = layers
        self.n_iter_train = int(100)
        self.n_iter_evaluate = int(100)
        self.expected_range = expected_range

    def train(self, training_set):
        for iteration in range(self.n_iter_train):
            x = self.normalize(next(training_set()).ravel())
            print(x)

    def evaluate(self, evaluation_set):
        for iteration in range(self.n_iter_evaluate):
            x = self.normalize(next(evaluation_set()).ravel())

    def normalize(self, values):
        min_val = self.expected_range[0]
        max_val = self.expected_range[1]
        scale_factor = max_val - min_val
        offset_factor = min_val
        return (values - offset_factor) / scale_factor - 0.5