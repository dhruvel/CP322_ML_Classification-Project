from testing import plot_accuracy_iterations, plot_accuracy_cost_threshold
from data.data_all import ionosphere_data, adult_data, diabetes_data, spambase_data
from utils import load_models

def plot(data, data_name):
    cost_change_threshold = 0
    max_iterations = 500
    learning_rates = [0.01, 0.05, 0.1, 0.2]

    plot_accuracy_iterations(
        data,
        data_name,
        learning_rates,
        cost_change_threshold=cost_change_threshold,
        max_iterations=max_iterations,
        test_split_ratio=0.8,
        print_Progress=True,
    )

def plot_all_accuracy_cost_threshold():
    cost_change_thresholds = [0.00001, 0.0001, 0.001]

    # ionosphere data
    plot_accuracy_cost_threshold(
        ionosphere_data,
        "Ionosphere",
        regularization_lambda=0.01,
        cost_change_thresholds=cost_change_thresholds,
        print_Progress=True,
    )

    # adult data
    plot_accuracy_cost_threshold(
        adult_data,
        "Adult",
        regularization_lambda=0.5,
        max_iterations=5000,
        test_split_ratio=0.85,
        cost_change_thresholds=cost_change_thresholds,
        print_Progress=True,
    )

    # diabetes data
    plot_accuracy_cost_threshold(
        diabetes_data,
        "Diabetes",
        regularization_lambda=0.03,
        max_iterations=5000,
        test_split_ratio=0.85,
        cost_change_thresholds=cost_change_thresholds,
        print_Progress=True,
    )

    # spambase data
    plot_accuracy_cost_threshold(
        spambase_data,
        "Spambase",
        regularization_lambda=0.01,
        test_split_ratio=0.9,
        cost_change_thresholds=cost_change_thresholds,
        print_Progress=True,
    )

'''# Load the dataset for testing size scaling
max_dataset_size = 2000  # Adjust as needed

# Call the size scaling plot function
plot_accuracy_size(model_instance, train_data, max_dataset_size)'''

if __name__ == '__main__':
    # plot(adult_data, "Adult")
    # plot(diabetes_data, "Diabetes")
    # plot(spambase_data, "Spambase")
    # plot_all_accuracy_cost_threshold()
    pass
