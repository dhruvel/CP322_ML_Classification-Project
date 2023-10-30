from enum import Enum
from model_interface import ModelInterface
from logistic_regression import LogisticRegression
from data.data_all import ionosphere_data
from utils import save_model, load_models
from testing import find_best_logistic_model

# Model Arg Enums
class ModelArg(Enum):
    LEARNING_RATE = 0
    REGULARIZATION_LAMBDA = 1
    COST_CHANGE_THRESHOLD = 2
    ITERATIONS = 3
    DATA_SIZE = 4
    ACCURACY = 5
    TRAINING_COST = 6
    TEST_COST = 7

def train_ionosphere_logistic_model(
        learning_rates=[0.001, 0.005, 0.01],
        regularization_lambdas=[0.01, 0.1, 0.2, 0.5],
        cost_change_threshold=0.00001,
        max_iterations=15000,
        test_split_ratio=0.95,
        max_cost_diff=0.2,
        max_cost=80,
        print_acc=True,
        model_file="ionosphere_models.csv",
):
    # Fit ionosphere data, trying different regularization lambdas, learning rates, and using 5-fold cross validation
    best_model, accuracy, test_cost = find_best_logistic_model(
        ionosphere_data,
        cost_change_threshold=cost_change_threshold,
        learning_rates=learning_rates,
        regularization_lambdas=regularization_lambdas,
        max_iterations=max_iterations,
        test_split_ratio=test_split_ratio,
        max_cost_diff=max_cost_diff,
        max_cost=max_cost,
        print_acc=print_acc,
        model_file=model_file,
    )

    if best_model is None:
        print("No model found based on given parameters")
        return None
        
    print("Ionosphere data")
    print("Best model learning rate and lambda: {}, {}".format(best_model.learning_rate, best_model.regularization_lambda))
    print("Best model test accuracy: {}".format(accuracy))
    print("Best model test cost: {}".format(test_cost))
    print("Best model iterations: {}".format(best_model.iterations))

    save_model(
        [
            "ionosphere",
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

    return best_model

def get_models_with_args(model_file, args, max_cost=0.1, model_type: ModelInterface=LogisticRegression):
    # Load models from file
    models = load_models(model_file, 8)
    
    # Find models with given args
    matching_models = []
    for model in models:
        match = True
        for arg, value in args.items():
            if model["args"][arg.value] != value:
                match = False
                break

        if match:
            matching_models.append(model)
    
    # Filter out models with cost above max_cost
    matching_models = [model for model in matching_models if abs(model["args"][ModelArg.TRAINING_COST.value] - model["args"][ModelArg.TEST_COST.value]) < max_cost]
    # Sort models by accuracy
    matching_models.sort(key=lambda x: x["args"][ModelArg.ACCURACY.value], reverse=True)

    return [(model_type.Load(args=x["args"], params=x["params"], b=x["b"]), x["args"]) for x in matching_models]

def get_top_model(model_file, dataset, model_type: ModelInterface=LogisticRegression):
    # Load models from file
    models = load_models(model_file, 8)
    # Find model from dataset
    models = [model for model in models if model["args"][0] == dataset]

    return model_type.Load(args=models[0]["args"], params=models[0]["params"], b=models[0]["b"])