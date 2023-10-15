from testing import plot_accuracy_iterations
from data.data_all import adult_data, ionosphere_data, diabetes_data, spambase_data
from utils import load_models

def plot(filename, data, data_name):
    models = load_models(filename, 5)
    k = 5
    cost_change_threshold = 0.00001
    max_iterations = 500
    learning_rates = [0.01, 0.05, 0.1, 0.2]

    plot_accuracy_iterations(
        data,
        data_name,
        models,
        learning_rates,
        k,
        cost_change_threshold=cost_change_threshold,
        max_iterations=max_iterations,
        test_split_ratio=0.8,
        print_acc=True,
        model_file=filename,
    )

'''# Load the dataset for testing size scaling
max_dataset_size = 2000  # Adjust as needed

# Call the size scaling plot function
plot_accuracy_size(model_instance, train_data, max_dataset_size)'''

if __name__ == '__main__':
  plot("adult_models.csv", adult_data, "Adult")
  plot("ionosphere_models.csv", ionosphere_data, "Ionosphere")
  plot("diabetes_models.csv", diabetes_data, "Diabetes")
  plot("spambase_models.csv", spambase_data, "Spambase")