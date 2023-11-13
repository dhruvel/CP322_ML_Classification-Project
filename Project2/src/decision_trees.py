import os
from sklearn.tree import DecisionTreeClassifier
from utils import train_imdb_classifier, find_classifier_accuracy, cross_validate_score, train_newsgroup_classifier
from model_utils import save_params, load_params
from imdb_data import imdb_test, imdb_train
from newsgroup_data import newsgroup_test, newsgroup_train
from plot import plot_accuracy_over_param

DECISION_TREES = '../models/decision_trees.json'

def train_save_decision_tree(train_func, params_key):
    # Make sure we're in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Train Decision Tree on IMDB data
    pipeline = train_func(DecisionTreeClassifier(), print_debug=True)
    clf = pipeline.named_steps['clf']
    save_params(clf.get_params(), DECISION_TREES, params_key)
  
def decision_tree_accuracy(test_data, train_data, train_function, params_key):
    # Make sure we're in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Load Decision Tree params
    params = load_params(DECISION_TREES)
    params = params[params_key]

    # Train and evaluate Decision Tree
    pipeline = train_function(DecisionTreeClassifier(**params), print_debug=False)
    clf = pipeline.named_steps['clf']
    print('Cross validated accuracy:', cross_validate_score(pipeline, train_data))
    print('Classifier accuracy on test:', find_classifier_accuracy(pipeline, test_data))
    print('Depth of tree:', clf.get_depth())

def decision_tree_plot(train_data, filename):
    # Make sure we're in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Plot Decision Tree accuracy over max_depth
    plot_accuracy_over_param(
        DecisionTreeClassifier(),
        train_data,
        param='min_samples_leaf',
        param_vals=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        file_name=filename,
        print_debug=False,
    )
if __name__ == '__main__':
    #train_save_decision_tree(train_imdb_classifier, 'imdb')
    #train_save_decision_tree(train_newsgroup_classifier, 'news')
    decision_tree_accuracy(newsgroup_test, newsgroup_train, train_newsgroup_classifier, 'news')
    print('------------------')
    #decision_tree_accuracy(imdb_test, imdb_train, train_imdb_classifier, 'imdb')
    #print('------------------')
    #decision_tree_plot(newsgroup_train, 'decision_min_samples_leaf_newsgroup')
    #decision_tree_plot(imdb_train, 'decision_tree_criterion_imdb')
