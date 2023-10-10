from typing import Callable, List
import numpy as np
from model_interface import ModelInterface

def split_data(data, ratio=0.9):
    train_size = int(len(data) * ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

def evaluate_acc(predictions, labels):
    return sum(predictions == labels) / len(labels)

def kfold_cross_validation(data, create_model: Callable[[], ModelInterface], k=5, training_threshold=0.1, max_iterations=10000):
    # Split data into k folds
    folds = []
    fold_size = int(len(data) / k)
    for i in range(k):
        folds.append(data[i * fold_size:(i + 1) * fold_size])

    # Train k models
    models: List[ModelInterface] = []
    for i in range(k):
        # Train model on k-1 folds
        train_data = np.concatenate(folds[:i] + folds[i + 1:])
        train_data = train_data[:, :-1]
        train_labels = train_data[:, -1]

        model = create_model()
        model.fit(train_data, train_labels, training_threshold=training_threshold, max_iterations=max_iterations)
        models.append(model)

    # Test k models
    accuracies = []
    for i in range(k):
        # Test model on 1 fold
        cv_data = folds[i]
        cv_labels = cv_data[:, -1]
        cv_data = cv_data[:, :-1]

        predictions = models[i].predict(cv_data)
        accuracies.append(evaluate_acc(predictions, cv_labels))
        
    return sum(accuracies) / len(accuracies)

def save_model(args, model: ModelInterface, filename: str):
    with open(filename, "a") as f:
        for arg in args:
            f.write(arg + ",")
        for param in model.params:
            f.write(str(param) + ",")
        f.write(str(model.b) + "\n")

def load_models(filename: str, arg_num: int):
    with open(filename, "r") as f:
        models = []
        for line in f:
            values = line.split(",")
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