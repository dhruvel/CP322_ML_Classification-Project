from testing import plot_accuracy_iterations
from data.data_all import adult_data
from utils import load_models

filename = "adult_models.csv"
models = load_models(filename, 5)
k = 5
cost_change_threshold = 0.00001
learning_rates = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

plot_accuracy_iterations(
    adult_data,
    models,
    learning_rates,
    k,
    cost_change_threshold=cost_change_threshold,
    max_iterations=1000,
    test_split_ratio=0.8,
    print_acc=True,
    model_file=filename,
)

'''# Load the dataset for testing size scaling
max_dataset_size = 2000  # Adjust as needed

# Call the size scaling plot function
plot_accuracy_size(model_instance, train_data, max_dataset_size)'''