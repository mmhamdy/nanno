import numpy as np

class Operation:
    """
    Base class for an operation
    """
    def __init__(self, params=None):
        if params:
            self.param = params

    def forward():
        raise NotImplementedError

    def backward():
        raise NotImplementedError


class Layer:
    """
    Base class for a layer
    """
    def __init__(self):
        pass