from abc import ABC, abstractmethod

import numpy as np

class Operation(ABC):
    """
    Base class for an operation
    """
    def __init__(self, params: np.ndarray) -> None:
        self.param = params
        
    @abstractmethod
    def forward(self, input_array: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    


class Layer(ABC):
    """
    Base class for a layer
    """
    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = in_features
        self.out_features = out_features

    @abstractmethod
    def setup(self):
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, input_array: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def backward(self, grad_array: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def init_params(self) -> None:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def params(self):
        raise NotImplementedError
    
    @params.setter
    @abstractmethod
    def params(self, value):
        raise NotImplementedError

