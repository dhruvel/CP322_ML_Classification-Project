import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from data_utils import split_data
from newsgroup_data import newsgroup_train, newsgroup_all
from imdb_data import imdb_train, imdb_all

"""
Functions in this file:
    - train_newsgroup_classifier(classifier, params=None, print_debug=False)
    - train_imdb_classifier(classifier, params=None, print_debug=False)
    - cross_validate_score(classifier, train_data, k=5)
    - find_classifier_accuracy(classifier, test_data)
    - test_newsgroup_data_splits(classifier, ratios=[0.25, 0.50, 0.75, 0.95], print_debug=False)
    - test_imdb_data_splits(classifier, ratios=[0.25, 0.50, 0.75, 0.95], print_debug=False)

Usage:
    1. Use train_newsgroup_classifier and train_imdb_classifier to train classifiers on the data.
        - If you pass down params, it will try each combination of parameters and return the best one. See examples below.
        - You can also use cross_validate_score to get the training score from cross validation for a specific data set.
    2. Use test_newsgroup_data_splits and test_imdb_data_splits to test the final accuracy of the classifiers on different data splits.
        - Final accuracy of a classifier can also be found using find_classifier_accuracy.
"""


def _train_classifier(classifier, train_data, print_debug=False) -> Pipeline:
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', classifier)
    ], verbose=print_debug)

    pipeline.fit(train_data.data, train_data.target)
    return pipeline

def _train_classifier_params(classifier, train_data, params, print_debug=False) -> Pipeline:
    # Create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', classifier)
    ], verbose=print_debug)

    # Append 'clf__' to each parameter name to target the classifier
    params = {f'clf__{k}': v for k, v in params.items()}

    # Run grid search to find best hyperparameters
    grid_search_clf = GridSearchCV(pipeline, params, n_jobs=-1, verbose=print_debug)
    grid_search_clf.fit(train_data.data, train_data.target)
    return grid_search_clf

"""
Trains the given classifier on the newsgroup data and returns the pipeline.

Parameters:
    classifier: The classifier to train
    print_debug: Whether to print debug information
Returns:
    (Pipeline) The sklearn pipeline

Example Usage:
    ## Ex 1: No hyperparameters
    from sklearn.naive_bayes import MultinomialNB
    clf = train_newsgroup_classifier(MultinomialNB(), print_debug=True)
    clf.predict(['God is love'])

    ## Ex 2: With hyperparameters
    # Different classifier parameters to try
    # This will test alpha=0.001 and alpha=0.0001
    params = {
        'alpha': (0.001, 0.0001)
    }
    clf = train_newsgroup_classifier(MultinomialNB(), print_debug=True, params=params)
    clf.predict(['God is love'])
    
    # Get the best parameters
    print(clf.best_params_)
    # Get all the results
    print(clf.cv_results_)
"""
def train_newsgroup_classifier(classifier, params=None, print_debug=False):
    if params is not None:
        return _train_classifier_params(classifier, newsgroup_train, params, print_debug)
    else:
        return _train_classifier(classifier, newsgroup_train, print_debug)

"""
Trains the given classifier on the IMDB data and returns the pipeline.

Parameters:
    classifier: The classifier to train
    print_debug: Whether to print debug information
    params: Hyperparameters to use for grid search
Returns:
    (Pipeline) The sklearn pipeline

Example Usage:
    ## Ex 1: No hyperparameters
    from sklearn.naive_bayes import MultinomialNB
    clf = train_imdb_classifier(MultinomialNB(), print_debug=True)
    clf.predict(['This movie was great!'])

    ## Ex 2: With hyperparameters
    # Different classifier parameters to try
    # This will test alpha=0.001 and alpha=0.0001
    params = {
        'alpha': (0.001, 0.0001)
    }
    clf = train_imdb_classifier(MultinomialNB(), print_debug=True, params=params)
    print(clf.predict(['This movie was great!']))
    
    # Get the best parameters
    print(clf.best_params_)
    # Get all the results
    print(clf.cv_results_)
"""
def train_imdb_classifier(classifier, params=None, print_debug=False):
    if params is not None:
        return _train_classifier_params(classifier, imdb_train, params, print_debug)
    else:
        return _train_classifier(classifier, imdb_train, print_debug)

"""
Return the training score from cross validation.

Example Usage:
    from sklearn.naive_bayes import MultinomialNB
    clf = train_imdb_classifier(MultinomialNB(), print_debug=True)
    print(cross_validate_score(clf, imdb_train))
"""
def cross_validate_score(classifier, train_data, k=5):
    kfold = KFold(n_splits=k, shuffle=True)
    scores = cross_val_score(classifier, train_data.data, train_data.target, cv=kfold)
    return np.mean(scores)

"""
Returns the accuracy of the classifier on the given test data.

Example Usage:
    from sklearn.naive_bayes import MultinomialNB
    from imdb_data import imdb_test
    clf = train_imdb_classifier(MultinomialNB(), print_debug=True)
    print(find_classifier_accuracy(clf, imdb_test))
"""
def find_classifier_accuracy(classifier, test_data):
    predictions = classifier.predict(test_data.data)
    return np.mean(predictions == test_data.target)

def _test_data_splits(data, classifier, ratios=[0.25, 0.50, 0.75, 0.95], print_debug=False):
    accuracies = []
    for ratio in ratios:
        # Split the data into train and test sets
        train, test = split_data(data, ratio=ratio)
        if print_debug:
            print("Ratio: {}, Train: {}, Test: {}".format(ratio, len(train), len(test)))

        # Train the classifier
        pipeline = _train_classifier(classifier, train, print_debug)

        # Test the classifier
        accuracy = find_classifier_accuracy(pipeline, test)
        accuracies.append(accuracy)

    return accuracies

"""
Returns the accuracies of the given classifier on the newsgroup data for the given ratios.
The data gets split into train and test sets using ratio for the training data and (1 - ratio) for the test data.
* To be used to test the final accuracies of a classifier, not for cross validation. *

Parameters:
    classifier: The classifier to test
    ratios: The ratios to split the data into train and test sets. Default: [0.25, 0.50, 0.75, 0.95]
    print_debug: Whether to print debug information

Returns:
    (List) The accuracies for each ratio

Example Usage:
    from sklearn.naive_bayes import MultinomialNB
    print(test_newsgroup_data_splits(MultinomialNB(), print_debug=True))
"""
def test_newsgroup_data_splits(classifier, ratios=[0.25, 0.50, 0.75, 0.95], print_debug=False):
    return _test_data_splits(newsgroup_all, classifier, ratios, print_debug)

"""
Returns the accuracies of the given classifier on the imdb data for the given ratios.
The data gets split into train and test sets using ratio for the training data and (1 - ratio) for the test data.
* To be used to test the final accuracies of a classifier, not for cross validation. *

Parameters:
    classifier: The classifier to test
    ratios: The ratios to split the data into train and test sets. Default: [0.25, 0.50, 0.75, 0.95]
    print_debug: Whether to print debug information

Returns:
    (List) The accuracies for each ratio

Example Usage:
    from sklearn.naive_bayes import MultinomialNB
    print(test_imdb_data_splits(MultinomialNB(), print_debug=True))
"""
def test_imdb_data_splits(classifier, ratios=[0.25, 0.50, 0.75, 0.95], print_debug=False):
    return _test_data_splits(imdb_all, classifier, ratios, print_debug)
