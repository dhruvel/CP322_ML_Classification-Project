from logistic_regression import LogisticRegression

# Load all data
from data.data_all import adult_data, ionosphere_data, wine_data, rice_data

# Fit ionosphere data
ionosphere_lr = LogisticRegression(learning_rate=0.01, regularization_lambda=0)
ionosphere_lr.fit(ionosphere_data[:, :-1], ionosphere_data[:, -1])

print(ionosphere_lr.params)
print(ionosphere_lr.b)
print(ionosphere_lr.cost)
print(ionosphere_lr.iterations)
