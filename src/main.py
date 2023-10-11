import numpy as np
from utils import save_model
from testing import find_best_logistic_model

# Models
from logistic_regression import LogisticRegression

# Load all data
from data.data_all import adult_data, ionosphere_data, wine_data, lung_cancer_data

# Fit ionosphere data, trying different regularization lambdas, learning rates, and using 5-fold cross validation
# threshold = 0.25
# best_model, accuracy, test_cost = find_best_logistic_model(
#     ionosphere_data,
#     training_threshold=threshold,
#     print_acc=True,
#     model_file="ionosphere_models.csv",
# )

# print("Ionosphere data")
# print("Best model learning rate and lambda: {}, {}".format(best_model.learning_rate, best_model.regularization_lambda))
# print("Best model test accuracy: {}".format(accuracy))
# print("Best model test cost: {}".format(test_cost))
# print("Best model iterations: {}".format(best_model.iterations))

# save_model(
#     [
#         "ionosphere",
#         best_model.learning_rate,
#         best_model.regularization_lambda,
#         threshold,
#         best_model.iterations,
#         accuracy,
#         test_cost,
#     ],
#     best_model,
#     "top_models.csv"
# )

# Fit adult data, trying different regularization lambdas, learning rates, and using 5-fold cross validation
threshold = 0.25
best_model, accuracy, test_cost = find_best_logistic_model(
    adult_data,
    training_threshold=threshold,
    learning_rates=[0.01, 0.05, 0.1, 0.2],
    regularization_lambdas=[0.1, 0.2, 0.5, 1, 2],
    max_iterations=1000,
    test_split_ratio=0.8,
    print_acc=True,
    model_file="adult_models.csv",
)

print("Adult data")
print("Best model learning rate and lambda: {}, {}".format(best_model.learning_rate, best_model.regularization_lambda))
print("Best model test accuracy: {}".format(accuracy))
print("Best model test cost: {}".format(test_cost))
print("Best model iterations: {}".format(best_model.iterations))

save_model(
    [
        "adult",
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

# # Fit wine data, trying different regularization lambdas, learning rates, and using 5-fold cross validation
# threshold = 0.25
# best_model, accuracy, test_cost = find_best_logistic_model(
#     wine_data,
#     training_threshold=threshold,
#     print_acc=True,
#     model_file="wine_models.csv",
# )

# print("Wine data")
# print("Best model learning rate and lambda: {}, {}".format(best_model.learning_rate, best_model.regularization_lambda))
# print("Best model test accuracy: {}".format(accuracy))
# print("Best model test cost: {}".format(test_cost))
# print("Best model iterations: {}".format(best_model.iterations))

# save_model(
#     [
#         "wine",
#         best_model.learning_rate,
#         best_model.regularization_lambda,
#         threshold,
#         best_model.iterations,
#         accuracy,
#         test_cost,
#     ],
#     best_model,
#     "top_models.csv"
# )

# # Fit lung cancer data, trying different regularization lambdas, learning rates, and using 5-fold cross validation
# threshold = 0.25
# best_model, accuracy, test_cost = find_best_logistic_model(
#     lung_cancer_data,
#     training_threshold=threshold,
#     print_acc=True,
#     model_file="lung_cancer_models.csv",
# )

# print("Lung Cancer data")
# print("Best model learning rate and lambda: {}, {}".format(best_model.learning_rate, best_model.regularization_lambda))
# print("Best model test accuracy: {}".format(accuracy))
# print("Best model test cost: {}".format(test_cost))
# print("Best model iterations: {}".format(best_model.iterations))

# save_model(
#     [
#         "lung_cancer",
#         best_model.learning_rate,
#         best_model.regularization_lambda,
#         threshold,
#         best_model.iterations,
#         accuracy,
#         test_cost,
#     ],
#     best_model,
#     "top_models.csv"
# )