from typing import Callable, List
import numpy as np
from model_interface import ModelInterface

def split_data(data, ratio=0.95):
    train_size = int(len(data) * ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

def evaluate_acc(predictions, labels):
    return sum(predictions == labels) / len(labels)

def kfold_cross_validation(
        data,
        create_model: Callable[[], ModelInterface],
        k=5,
        training_threshold=0.1,
        max_iterations=10000,
        print_acc=False,
        model_file=""
    ):
    # Shuffle data
    np.random.shuffle(data)

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

def save_model(args, model: ModelInterface, filename: str):
    with open("models/" + filename, "a") as f:
        for arg in args:
            f.write(str(arg) + ",")
        for param in model.params:
            f.write(str(param) + ",")
        f.write(str(model.b[0]) + "\n")

def load_models(filename: str, arg_num: int):
    with open("models/" + filename, "r") as f:
        models = []
        for line in f:
            values = line.split(",")
            if len(values) <= 1 or values[0].startswith("#"):
                continue

            args = float(values[:arg_num])
            params = float(values[arg_num:-1])
            b = float(values[-1])
            
            model = {
                "args": args,
                "params": params,
                "b": b
            }
            models.append(model)

    return models