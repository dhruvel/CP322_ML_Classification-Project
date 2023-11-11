import matplotlib.pyplot as plt

from utils import _train_classifier_params

"""
Plots the accuracy of the given classifier over the given parameter.

Parameters:
    classifier: The classifier to train
    train_data: The training data
    param: The name of the parameter variable to plot
    param_vals: The values of the parameter to test and plot
    file_name: The name of the file to save the plot to. If not given, the file name will be <classifier_name>-<param>_accuracy.png
    print_debug: Whether to print debug information

Returns:
    None

Usage:
    from newsgroup_data import newsgroup_train
    from sklearn.naive_bayes import MultinomialNB
    plot_accuracy_over_param(
        MultinomialNB(),
        newsgroup_train,
        param='alpha',
        param_vals=[0.1, 0.5, 1.0, 1.5, 2.0],
        print_debug=True,
    )

"""
def plot_accuracy_over_param(
        classifier,
        train_data,
        param: str,
        param_vals: list,
        file_name=None,
        print_debug=False
    ):
    # Train classifier with different values of param
    params = {
        param: param_vals,
    }
    clf = _train_classifier_params(classifier, train_data, params, print_debug)

    # Get accuracies
    accuracies = clf.cv_results_['mean_test_score']

    # Plot accuracies
    plt.figure()
    plt.plot(param_vals, accuracies)
    plt.xlabel(param)
    plt.ylabel("Accuracy")

    if file_name is None:
        file_name = f"{classifier.__class__.__name__}-{param}_accuracy"
    plt.savefig(f"../plots/{file_name}.png")

from newsgroup_data import newsgroup_train
from sklearn.naive_bayes import MultinomialNB
plot_accuracy_over_param(
    MultinomialNB(),
    newsgroup_train,
    'alpha',
    [0.1, 0.5, 1.0, 1.5, 2.0],
    print_debug=True,
)