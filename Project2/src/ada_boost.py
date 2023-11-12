import utils
from sklearn.ensemble import AdaBoostClassifier
from newsgroup_data import newsgroup_test
import matplotlib.pyplot as plt

# from imdb_data import imdb_test

# Train the newsgroup classifier
clf_1 = utils.train_newsgroup_classifier(AdaBoostClassifier(n_estimators=100))

# Train the IMDB classifier
# clf_2 = utils.train_imdb_classifier(AdaBoostClassifier(n_estimators=100))

# Predict the category of a movie review
# print(clf_2.predict(['This movie was great!']))

# Predict the category of a newsgroup post
print(clf_1.predict(['God is love']))

print(utils.find_classifier_accuracy(clf_1, newsgroup_test))
# print(utils.find_classifier_accuracy(clf_2, imdb_test))
