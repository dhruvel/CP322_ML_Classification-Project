from typing import Callable, List
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from model_interface import ModelInterface
from logistic_regression import LogisticRegression
from utils import split_data, evaluate_acc, save_model

def kfold_cross_validation(
        data,
        create_model: Callable[[], ModelInterface],
        k=5,
        cost_change_threshold=0.0001,
        max_iterations=15000,
        max_cost=np.Inf,
        print_acc=False,
        model_file="",
    ):
    # Split data into k folds
    folds = []
    fold_size = int(len(data) / k)
    for i in range(k):
        folds.append(data[i * fold_size:(i + 1) * fold_size])

    # Train k models in parallel using pools
    pool = mp.Pool(5)
    results = []
    for i in range(k):
        train_data = np.concatenate(folds[:i] + folds[i + 1:])
        model = create_model()
        results.append(pool.apply_async(
            model.fit,
            (train_data[:, :-1], train_data[:, -1], cost_change_threshold, max_iterations)
        ))
    pool.close()
    pool.join()

    accuracies = []
    cost_diffs = []
    best_acc = 0
    best_cost = np.Inf
    best_cost_model: ModelInterface = None
    accurate_model: ModelInterface = None
    for i in range(len(results)):
        # Test models on remaining fold with cross-validation data
        model = results[i].get()
        predicted, test_cost = model.predict(folds[i][:, :-1], folds[i][:, -1])
        accuracy = evaluate_acc(predicted, folds[i][:, -1])
        accuracies.append(accuracy)

        if print_acc:
            print("Fold {}: accuracy {}, cost {}, test cost: {}".format(i, accuracies[-1], model.cost, test_cost))

        if model_file != "":
            save_model(
                [
                    model.learning_rate,
                    model.regularization_lambda,
                    cost_change_threshold,
                    model.iterations,
                    train_data.shape[0],
                    accuracy,
                    model.cost,
                    test_cost,
                ],
                model,
                model_file
            )

        if model.cost > max_cost:
            continue

        # Keep track of best models
        cost_diffs.append(abs(model.cost - test_cost))
        if cost_diffs[-1] < best_cost:
            best_cost = cost_diffs[-1]
            least_cost_model = model

        if accuracies[-1] > best_acc:
            best_acc = accuracies[-1]
            accurate_model = model
    
    # Return average accuracy and best accuracy model
    if accurate_model is None:
        return 0, None, np.Inf, None
    return sum(accuracies) / len(accuracies), accurate_model, sum(cost_diffs) / len(cost_diffs), least_cost_model

def find_best_logistic_model(
        data,
        learning_rates=[0.005, 0.01, 0.05, 0.1],
        regularization_lambdas=[0, 0.1, 0.5, 1, 2],
        k=5,
        cost_change_threshold=0.0001,
        max_iterations=15000,
        max_cost_diff=0.1,
        max_cost=5,
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
                cost_change_threshold=cost_change_threshold,
                max_iterations=max_iterations,
                max_cost=max_cost,
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
            
            if kfold_cost_diff < max_cost_diff and kfold_cost_model is not None:
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
    if best_model is None:
        return None, 0, np.Inf
    
    predicted, test_cost = best_model.predict(test_data[:, :-1], test_data[:, -1])
    accuracy = evaluate_acc(predicted, test_data[:, -1])
    return best_model, accuracy, test_cost

def plot_accuracy_iterations(
        data,
        learning_rates=[0.005, 0.01, 0.05, 0.1],
        k=5,
        training_threshold=0.1,
        max_iterations=1000,
        test_split_ratio=0.80,
        print_acc=False,
        model_file="",
    ):

    plt.figure()

    np.random.shuffle(data)
    train_data, test_data = split_data(data, ratio=test_split_ratio)

    # Initialize lists to store iteration and accuracy values
    iteration_values = []
    accuracy_values = []

    for learning_rate in learning_rates:
        create_model = lambda: LogisticRegression(
            learning_rate=learning_rate,
            regularization_lambda=0,
        )
        for iteration in range(max_iterations):
            # Fit model using 5-fold cross validation
            kfold_acc, _, _, _ = kfold_cross_validation(
                train_data,
                create_model,
                k=k,
                training_threshold=training_threshold,
                max_iterations=iteration + 1,
                model_file=model_file,
                print_acc=print_acc
            )
            iteration_values.append(iteration + 1)
            accuracy_values.append(kfold_acc)

        # Plot accuracy vs. iterations for this learning rate
        plt.plot(iteration_values, accuracy_values, marker='o', label=f"LR: {learning_rate}")

    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Iterations for Different Learning Rates")
    plt.legend()
    plt.show()


def plot_accuracy_size():
    pass