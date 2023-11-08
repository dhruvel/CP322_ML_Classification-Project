import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

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

"""
Trains the given classifier on the newsgroup data and returns the pipeline.

Parameters:
    classifier: The classifier to train
    print_debug: Whether to print debug information
Returns:
    (Pipeline) The sklearn pipeline

Example Usage:
    from sklearn.naive_bayes import MultinomialNB
    clf = train_newsgroup_classifier(MultinomialNB(), print_debug=True)
    clf.predict(['God is love'])
"""
def train_newsgroup_classifier(classifier, print_debug=False):
    return _train_classifier(classifier, newsgroup_train, print_debug)

"""
Trains the given classifier on the IMDB data and returns the pipeline.

Parameters:
    classifier: The classifier to train
    print_debug: Whether to print debug information
Returns:
    (Pipeline) The sklearn pipeline

Example Usage:
    from sklearn.naive_bayes import MultinomialNB
    clf = train_imdb_classifier(MultinomialNB(), print_debug=True)
    clf.predict(['This movie was great!'])
"""
def train_imdb_classifier(classifier, print_debug=False):
    return _train_classifier(classifier, imdb_train, print_debug)

"""
Example Usage:
    from sklearn.naive_bayes import MultinomialNB
    clf = train_imdb_classifier(MultinomialNB(), print_debug=True)
    print(cross_validate_score(clf, imdb_train))
"""
def cross_validate_score(classifier, train_data, k=5):
    kfold = KFold(n_splits=k, shuffle=True)
    scores = cross_val_score(classifier, train_data.data, train_data.target, cv=kfold)
    return np.mean(scores)
