from abc import ABC, abstractmethod
import numpy as np
from typing import TypeVar

T = TypeVar('T', bound='ModelInterface')

class ModelInterface(ABC):
    @abstractmethod
    def fit(self, train_data, train_labels, training_threshold=0.1, max_iterations=np.Inf, print_cost=False) -> T:
        pass

    @abstractmethod
    def predict(self, test_data, test_labels=None):
        pass

    @abstractmethod
    def Load(args, params, b) -> T:
        pass