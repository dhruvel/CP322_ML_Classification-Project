from testing import plot_accuracy_iterations
from data.data_all import adult_data, ionosphere_data, diabetes_data, spambase_data
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

'''# Load the dataset for testing size scaling
max_dataset_size = 2000  # Adjust as needed

# Call the size scaling plot function
plot_accuracy_size(model_instance, train_data, max_dataset_size)'''

if __name__ == '__main__':
    # plot(adult_data, "Adult")
    plot(ionosphere_data, "Ionosphere")
#   plot(diabetes_data, "Diabetes")
#   plot(spambase_data, "Spambase")