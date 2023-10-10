from utils import split_data, evaluate_acc, kfold_cross_validation
# Models
from logistic_regression import LogisticRegression

# Load all data
from data.data_all import adult_data, ionosphere_data, wine_data, rice_data

# Fit ionosphere data
ionosphere_lr = LogisticRegression(learning_rate=0.01, regularization_lambda=0)
ionosphere_train, ionosphere_cv, ionosphere_test = split_data(ionosphere_data)
ionosphere_lr.fit(ionosphere_train[:, :-1], ionosphere_train[:, -1], max_iterations=10000)

print(ionosphere_lr.params)
print(ionosphere_lr.b)
print(ionosphere_lr.cost)
print(ionosphere_lr.iterations)

# Predict ionosphere data
ionosphere_predictions = ionosphere_lr.predict(ionosphere_test[:, :-1])
print(evaluate_acc(ionosphere_predictions, ionosphere_test[:, -1]))
