from typing import Callable, List
import numpy as np
from model_interface import ModelInterface
from logistic_regression import LogisticRegression
from utils import split_data, evaluate_acc, save_model

def kfold_cross_validation(
        data,
        create_model: Callable[[], ModelInterface],
        k=5,
        training_threshold=0.1,
        max_iterations=15000,
        print_acc=False,
        model_file="",
    ):
    # Split data into k folds
    folds = []
    fold_size = int(len(data) / k)
    for i in range(k):
        folds.append(data[i * fold_size:(i + 1) * fold_size])

    # Train k models
    accuracies = []
    cost_diffs = []
    accurate_model = None
    least_cost_model = None
    best_acc = 0
    best_cost = np.Inf
    for i in range(k):
        # Train model on k-1 folds
        train_data = np.concatenate(folds[:i] + folds[i + 1:])
        train_labels = train_data[:, -1]
        train_data = train_data[:, :-1]

        model = create_model()
        model.fit(train_data, train_labels, training_threshold=training_threshold, max_iterations=max_iterations)

        # Test model on remaining fold with cross-validation data
        cv_data = folds[i]
        cv_labels = cv_data[:, -1]
        cv_data = cv_data[:, :-1]

        predictions, test_cost = model.predict(cv_data, cv_labels)
        accuracies.append(evaluate_acc(predictions, cv_labels))

        if print_acc:
            print("Fold {}: accuracy {}, cost {}, test cost: {}".format(i, accuracies[-1], model.cost, test_cost))

        if model_file != "":
            save_model(
                [
                    model.learning_rate,
                    model.regularization_lambda,
                    training_threshold,
                    max_iterations,
                    train_data.shape[0],
                    accuracies[-1],
                    model.cost,
                    test_cost,
                ],
                model,
                model_file
            )

        # Keep track of best models
        cost_diffs.append(model.cost - test_cost)
        if cost_diffs[-1] < best_cost:
            best_cost = cost_diffs[-1]
            least_cost_model = model

        if accuracies[-1] > best_acc:
            best_acc = accuracies[-1]
            accurate_model = model
    
    # Return average accuracy and best accuracy model
    return sum(accuracies) / len(accuracies), accurate_model, sum(cost_diffs) / len(cost_diffs), least_cost_model

def find_best_logistic_model(
        data,
        learning_rates=[0.005, 0.01, 0.05, 0.1],
        regularization_lambdas=[0, 0.1, 0.5, 1, 2],
        k=5,
        training_threshold=0.1,
        max_iterations=15000,
        max_cost=0.1,
        test_split_ratio=0.95,
        print_acc=False,
        model_file="",
    ):
    # With default settings, it will train 4 * 5 = 20 models, each folded 5 times,
    # for a total of 100 models trained. This can take a while...
    np.random.shuffle(data)
    train_data, test_data = split_data(data, ratio=test_split_ratio)

    best_cost_models: List[{int, ModelInterface}] = []
    for learning_rate in learning_rates:
        for regularization_lambda in regularization_lambdas:
            # Create model
            create_model = lambda: LogisticRegression(
                learning_rate=learning_rate,
                regularization_lambda=regularization_lambda,
            )

            # Fit model using 5-fold cross validation
            kfold_acc, kfold_model, kfold_cost_diff, kfold_cost_model = kfold_cross_validation(
                train_data,
                create_model,
                k=k,
                training_threshold=training_threshold,
                max_iterations=max_iterations,
                model_file=model_file,
                print_acc=print_acc,
            )
            
            if print_acc:
                print("Learning rate: {}, Lambda: {}, Accuracy: {}, Cost Diff: {}".format(
                        learning_rate,
                        regularization_lambda,
                        kfold_acc,
                        kfold_cost_diff
                    ))
            
            if kfold_cost_diff < max_cost:
                best_cost_models.append((kfold_acc, kfold_cost_model))
    
    # Find highest accuracy model among those with lowest cost difference
    # between training cost and cross-validation cost
    best_acc = 0
    best_model: ModelInterface = None
    for acc, model in best_cost_models:
        if acc > best_acc:
            best_acc = acc
            best_model = model

    # Find accuracy of best model on test data
    predicted, test_cost = best_model.predict(test_data[:, :-1], test_data[:, -1])
    accuracy = evaluate_acc(predicted, test_data[:, -1])
    return best_model, accuracy, test_cost