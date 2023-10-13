import numpy as np
from model_interface import ModelInterface

def split_data(data, ratio=0.95):
    train_size = int(len(data) * ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

def evaluate_acc(predictions, labels):
    return sum(predictions == labels) / len(labels)

def save_model(args, model: ModelInterface, filename: str):
    with open("../models/" + filename, "a") as f:
        for arg in args:
            f.write(str(arg) + ",")
        for param in model.params:
            f.write(str(param) + ",")
        f.write(str(model.b[0]) + "\n")

def load_models(filename: str, arg_num: int):
    with open("../models/" + filename, "r") as f:
        models = {}
        for line in f:
            values = line.split(",")
            if len(values) <= 1 or values[0].startswith("#"):
                continue

            args = [float(value) for value in values[:arg_num]]
            params = [float(value) for value in values[arg_num:-1]]
            b = float(values[-1])
            
            model = {
                "args": args,
                "params": params,
                "b": b
            }
            key = tuple(args)
            if key in models:
                models[key].append(model)
            else:
                models[key] = [model]
    return models

# Search for all models with matching values, then find the most accurate one
def find_model(
        learning_rate,
        regularization_lambda,
        cost_change_threshold, 
        max_iterations,
        training_points,
        models
    ):
    model_key = (
        learning_rate, 
        regularization_lambda,
        cost_change_threshold,
        max_iterations,
        training_points,
    )
    if model_key not in models:
        return None
    
    potential_models = models[model_key]
    best_model = None
    best_accuracy = 0

    for model in potential_models:
        # Assuming accuracy is the first value in model['params']
        accuracy = model['params'][0]
        if accuracy > best_accuracy:
            best_model = model
            best_accuracy = accuracy

    return best_model

