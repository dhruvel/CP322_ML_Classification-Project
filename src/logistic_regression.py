import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate, regularization_lambda):
        self.learning_rate = learning_rate
        self.regularization_lambda = regularization_lambda

    def _sigmoid(self, x: np.ndarray):
        model_pred = np.dot(self.params, x) + self.b
        return 1 / (1 + np.exp(-model_pred))
    
    def _cost(self, y: np.ndarray, y_pred: np.ndarray):
        error = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        regularization = (self.regularization_lambda / 2 * len(self.params)) * np.sum(self.params ** 2)
        return error + regularization
    
    def _derivative_cost(self, param, param_x: np.ndarray, y: np.ndarray, y_pred: np.ndarray):
        return np.mean((y_pred - y) * param_x) + ((self.regularization_lambda * param) / len(self.params))

    def fit(self, train_data, train_labels, training_threshold=0.1, max_iterations=20000, print_cost=False):
        self.params = np.random.rand(train_data.shape[1])
        self.b = np.random.rand(1)

        self.cost = np.Inf
        self.iterations = 0
        while self.cost > training_threshold and self.iterations < max_iterations:
            self.iterations += 1
            if print_cost and self.iterations % 200 == 0:
                print(self.cost)

            y_pred = np.zeros(train_data.shape[0])
            for i in range(train_data.shape[0]):
                y_pred[i] = self._sigmoid(train_data[i])

            self.cost = self._cost(train_labels, y_pred)

            for i in range(len(self.params)):
                x = train_data[:, i]
                self.params[i] -= self.learning_rate * self._derivative_cost(self.params[i], x, train_labels, y_pred)
                self.b -= self.learning_rate * self._derivative_cost(0, 1, train_labels, y_pred)

    def predict(self, test_data):
        pass