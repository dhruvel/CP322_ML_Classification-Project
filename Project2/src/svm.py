from sklearn.svm import LinearSVC
from utils import *
from data_utils import split_data
from imdb_data import imdb_test 
from plot import *

# Initialize the SVM Classifiers
svm_classifier = LinearSVC()

# IMBD
# Best results found for IMBD with this C = .55
tuned_svm_classifier = LinearSVC(C=0.55)

default_pipeline = train_imdb_classifier(svm_classifier, print_debug=True)
tuned_pipeline = train_imdb_classifier(tuned_svm_classifier, print_debug=True)

pipelines = {"Default Pipeline": default_pipeline, "Tuned Pipeline": tuned_pipeline}
classifiers = {"Default Classifier": svm_classifier, "Tuned Classifier": tuned_svm_classifier}

ratios = [0.25, 0.50, 0.75, 0.95]

for name, pipeline in pipelines.items():
    # Perform k-fold cross-validation
    k_fold_score = cross_validate_score(pipeline, imdb_test, k=5)
    print(f"K-Fold Cross-Validation Score for {name}: {k_fold_score}")

for name, classifier in classifiers.items():
    # Test the classifier with different data splits
    split_test_results = test_imdb_data_splits(classifier, ratios=ratios, print_debug=True)
    for ratio, accuracy in zip(ratios, split_test_results):
        print(f"{name} - Ratio: {ratio}, Accuracy: {accuracy}")


# 20 NEWS GROUP
# Best results found for NEWS GROUP with C = 0.47
tuned_svm_classifier_20 = LinearSVC(C=0.47)

default_svm_pipeline_20 = train_newsgroup_classifier(svm_classifier, print_debug=True)
tuned_pipeline_20 = train_newsgroup_classifier(svm_classifier, print_debug=True)

pipelines_20 = {"Default Pipeline": default_svm_pipeline_20, "Tuned Pipeline": tuned_pipeline_20}
classifiers_20 = {"Default Classifier": svm_classifier, "Tuned Classifier": tuned_svm_classifier_20}

ratios = [0.25, 0.50, 0.75, 0.95]

for name, pipeline in pipelines_20.items():
    k_fold_score = cross_validate_score(pipeline, imdb_test, k=5)
    print(f"K-Fold Cross-Validation Score for {name}: {k_fold_score}")

for name, classifier in classifiers_20.items():
    split_test_results = test_imdb_data_splits(classifier, ratios=ratios, print_debug=True)
    for ratio, accuracy in zip(ratios, split_test_results):
        print(f"{name} - Ratio: {ratio}, Accuracy: {accuracy}")

def plot_accuracy_for_categorical_param(classifier, train_data, param, param_vals, file_name=None, print_debug=False):
    accuracies = []
    for val in param_vals:
        if param == 'penalty' and val == 'l1':
            classifier.set_params(**{param: val, 'dual': False})
        else:
            classifier.set_params(**{param: val})

        # Train and evaluate
        pipeline = train_imdb_classifier(classifier, print_debug=print_debug)  
        accuracy = cross_validate_score(pipeline, train_data, k=5)
        accuracies.append(accuracy)

    # Plot
    plt.figure()
    plt.bar(param_vals, accuracies)
    plt.xlabel(param)
    plt.ylabel("Accuracy")

    if file_name is None:
        file_name = f"{classifier.__class__.__name__}-{param}_accuracy"
    plt.savefig(f"../plots/{file_name}.png")

c_values = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25]
tol_values = [1e-4, 1e-3, 1e-2]
max_iter_values = [1000, 5000, 10000]
penalty_values = ['l1', 'l2']  

plot_accuracy_over_param(
    LinearSVC(),
    imdb_train,
    "C",
    c_values,
    file_name="IMDb_C_Accuracy",
    print_debug=True
)

plot_accuracy_over_param(
    LinearSVC(dual=False),  
    newsgroup_train,
    "tol",
    tol_values,
    file_name="newsgroup_Tol_Accuracy",
    print_debug=True
)

plot_accuracy_over_param(
    LinearSVC(),
    newsgroup_train,
    "max_iter",
    max_iter_values,
    file_name="newsgroup_Max_Iter_Accuracy",
    print_debug=True
)

plot_accuracy_for_categorical_param(
    LinearSVC(),
    newsgroup_train,
    "penalty",
    ['l1', 'l2'],
    file_name="newsgroup_Penalty_Accuracy",
    print_debug=True
)

