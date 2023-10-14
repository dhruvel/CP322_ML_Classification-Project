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
        models = []
        for line in f:
            values = line.split(",")
            if len(values) <= 1 or values[0].startswith("#"):
                continue

            args = np.array([float(v) for v in values[:arg_num]])
            params = np.array([float(v) for v in values[arg_num:-1]])
            b = float(values[-1])
            
            model = {
                "args": args,
                "params": params,
                "b": b
            }
            models.append(model)

    return models