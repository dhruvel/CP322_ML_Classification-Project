import numpy as np
from utils import save_model
from testing import find_best_logistic_model

# Models
from logistic_regression import LogisticRegression

# Load all data
from data.data_all import adult_data, ionosphere_data, wine_data, rice_data

# Fit ionosphere data
# ionosphere_train, ionosphere_test = split_data(ionosphere_data)
# ionosphere_model = LogisticRegression(learning_rate=0.01, regularization_lambda=0.15)
# ionosphere_model.fit(ionosphere_train[:, :-1], ionosphere_train[:, -1], training_threshold=0.1, max_iterations=15000)
# # Find accuracy of model on test data
# predicted = ionosphere_model.predict(ionosphere_test[:, :-1])
# accuracy = evaluate_acc(predicted, ionosphere_test[:, -1])
# print("Ionosphere accuracy: {}, cost: {}, test cost: {}".format(accuracy, ionosphere_model.cost, 0))

# Fit ionosphere data, trying different regularization lambdas, learning rates, and using 5-fold cross validation
threshold = 20
best_model, accuracy, test_cost = find_best_logistic_model(
    ionosphere_data,
    training_threshold=threshold,
    print_acc=True,
    model_file="ionosphere_models.csv",
)

print("Ionosphere data")
print("Best accuracy model learning rate and lambda: {}, {}".format(best_model.learning_rate, best_model.regularization_lambda))
print("Best accuracy model accuracy: {}".format(accuracy))
print("Best accuracy model params: {}".format(best_model.params))
print("Best accuracy model b: {}".format(best_model.b))
print("Best accuracy model cost difference: {}".format(test_cost))
print("Best accuracy model iterations: {}".format(best_model.iterations))

save_model(
    [
        "ionosphere",
        best_model.learning_rate,
        best_model.regularization_lambda,
        threshold,
        best_model.iterations,
        accuracy,
        test_cost,
    ],
    best_model,
    "top_models.csv"
)