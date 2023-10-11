import numpy as np
from utils import split_data, evaluate_acc, kfold_cross_validation

# Models
from logistic_regression import LogisticRegression

# Load all data
from data.data_all import adult_data, ionosphere_data, wine_data, rice_data

# Fit ionosphere data
ionosphere_train, ionosphere_test = split_data(ionosphere_data)
ionosphere_model = LogisticRegression(learning_rate=0.01, regularization_lambda=0.15)
ionosphere_model.fit(ionosphere_train[:, :-1], ionosphere_train[:, -1], training_threshold=0.1, max_iterations=15000)
# Find accuracy of model on test data
predicted = ionosphere_model.predict(ionosphere_test[:, :-1])
accuracy = evaluate_acc(predicted, ionosphere_test[:, -1])
print("Ionosphere accuracy: {}, cost: {}, test cost: {}".format(accuracy, ionosphere_model.cost, 0))

# Fit ionosphere data, trying different regularization lambdas, learning rates, and using 5-fold cross validation
ionosphere_train, ionosphere_test = split_data(ionosphere_data)
best_acc = 0
best_cost = np.Inf
accurate_model = None
least_cost_model = None
for learning_rate in [0.005, 0.01, 0.05, 0.1]:
    for regularization_lambda in [0, 0.1, 0.2, 0.5, 1]:
        # Create model using current learning rate and lambda
        create_lr = lambda: LogisticRegression(learning_rate=learning_rate, regularization_lambda=regularization_lambda)
        # Fit model using 5-fold cross validation
        kfold_acc, kfold_model, kfold_cost, kfold_cost_model = kfold_cross_validation(
            ionosphere_train,
            create_lr,
            max_iterations=15000,
            model_file="ionosphere_models_test_cost.csv"
        )
        
        print("Learning rate: {}, Lambda: {}, Accuracy: {}".format(learning_rate, regularization_lambda, kfold_acc))
        if kfold_acc > best_acc:
            best_acc = kfold_acc
            accurate_model = kfold_model
        
        if kfold_cost < best_cost:
            best_cost = kfold_cost
            least_cost_model = kfold_cost_model

# Find accuracy of best model on test data
print("Ionosphere data")

predicted = accurate_model.predict(ionosphere_test[:, :-1])
accuracy = evaluate_acc(predicted, ionosphere_test[:, -1])
print("Best accuracy model learning rate and lambda: {}, {}".format(accurate_model.learning_rate, accurate_model.regularization_lambda))
print("Best accuracy model accuracy: {}".format(accuracy))
print("Best accuracy model params: {}".format(accurate_model.params))
print("Best accuracy model b: {}".format(accurate_model.b))
print("Best accuracy model cost: {}".format(accurate_model.cost))
print("Best accuracy model iterations: {}".format(accurate_model.iterations))

predicted = least_cost_model.predict(ionosphere_test[:, :-1])
accuracy = evaluate_acc(predicted, ionosphere_test[:, -1])
print("Best cost model learning rate and lambda: {}, {}".format(least_cost_model.learning_rate, least_cost_model.regularization_lambda))
print("Best cost model accuracy: {}".format(least_cost_model.accuracy))
print("Best cost model params: {}".format(least_cost_model.params))
print("Best cost model b: {}".format(least_cost_model.b))
print("Best cost model cost: {}".format(least_cost_model.cost))
print("Best cost model iterations: {}".format(least_cost_model.iterations))
