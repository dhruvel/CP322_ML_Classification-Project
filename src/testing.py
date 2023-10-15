from typing import Callable, List
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from model_interface import ModelInterface
from logistic_regression import LogisticRegression
from utils import split_data, evaluate_acc, save_model, find_model

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
    least_cost_model: ModelInterface = None
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
        data_name,
        learning_rates=[0.005, 0.01, 0.05, 0.1],
        cost_change_threshold=0.0001,
        max_iterations=1000,
        test_split_ratio=0.80,
        print_Progress=False,
    ):
    regularization_lambda = 0
    plt.figure()

    # Initialize training data
    np.random.shuffle(data)
    train_data, test_data = split_data(data, ratio=test_split_ratio)

    colors = ['r', 'g', 'b', 'c', 'm']

    for i, learning_rate in enumerate(learning_rates):
        # Initialize lists to store iteration and accuracy values
        model = LogisticRegression(
            learning_rate=learning_rate,
            regularization_lambda=regularization_lambda,
        )

        model.fit(
            train_data[:, :-1],
            train_data[:, -1],
            cost_change_threshold=cost_change_threshold,
            max_iterations=max_iterations,
            test_data=test_data[:, :-1],
            test_labels=test_data[:, -1],
        )

        if print_Progress:
            print("Learning rate: {}, Lambda: {}, Accuracy: {}".format(
                    learning_rate,
                    regularization_lambda,
                    model.accuracies[-1]
                ))

        # Plot accuracy vs. iterations for this learning rate
        plt.plot(range(0, max_iterations), model.accuracies, marker='.', label=f"LR: {learning_rate}", c=colors[i])

    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs. Iterations for Different Learning Rates, {data_name} Data")
    plt.legend()
    plt.ylim(0, 1)  # Set the y-axis limits
    plt.savefig("accuracy_vs_iterations.png")

def plot_accuracy_size():
    pass