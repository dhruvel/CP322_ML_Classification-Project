import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from newsgroup_data import newsgroup_train
from imdb_data import imdb_train

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
def train_newsgroup_classifier(classifier, print_debug=False, params=None):
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
def train_imdb_classifier(classifier, print_debug=False, params=None):
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
