from testing import find_best_logistic_model
from utils import save_model

# Load all data
from data.data_all import adult_data, ionosphere_data, wine_data, lung_cancer_data

# Fit adult data, trying different regularization lambdas, learning rates, and using 5-fold cross validation
cost_change_threshold = 0.00001
best_model, accuracy, test_cost = find_best_logistic_model(
    adult_data,
    cost_change_threshold=cost_change_threshold,
    learning_rates=[0.001, 0.01, 0.1],
    regularization_lambdas=[0.5, 0.7, 1, 1.2],
    max_iterations=4000,
    test_split_ratio=0.85,
    max_cost_diff=0.1,
    print_acc=True,
    model_file="adult_models.csv",
)

if best_model is None:
    print("No model found based on given parameters")
    exit()

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
        cost_change_threshold,
        best_model.iterations,
        accuracy,
        test_cost,
    ],
    best_model,
    "top_models.csv"
)
