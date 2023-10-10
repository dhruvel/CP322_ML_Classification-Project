from abc import ABC, abstractmethod
import numpy as np

class ModelInterface(ABC):
    @abstractmethod
    def fit(self, train_data, train_labels, training_threshold=0.1, max_iterations=np.Inf, print_cost=False):
        pass

    @abstractmethod
    def predict(self, test_data):
        pass

    @abstractmethod
    def load(self, params, b):
        pass