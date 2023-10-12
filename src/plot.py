from testing import plot_accuracy_iterations
from data.data_all import adult_data

k = 5
threshold = 0.25
learning_rates = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

plot_accuracy_iterations(
  adult_data,
  learning_rates,
  k,
  training_threshold=threshold,
  max_iterations=1000,
  test_split_ratio=0.8,
  print_acc=False,
  model_file="",
)

'''# Load the dataset for testing size scaling
max_dataset_size = 2000  # Adjust as needed

# Call the size scaling plot function
plot_accuracy_size(model_instance, train_data, max_dataset_size)'''