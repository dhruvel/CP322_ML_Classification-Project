import utils
from sklearn.ensemble import AdaBoostClassifier
from newsgroup_data import newsgroup_train,newsgroup_test
from imdb_data import imdb_test,imdb_train
from plot import plot_accuracy_over_param
from utils import find_classifier_accuracy, train_imdb_classifier, train_newsgroup_classifier

# AdaBoost over different number of estimators
# plot_accuracy_over_param(AdaBoostClassifier(),
#  newsgroup_train,
#  'n_estimators',
#  [50,100,150,200],
#  file_name="AdaBoostNewsgroup",
#  print_debug=True
#  )

# plot_accuracy_over_param(AdaBoostClassifier(),
#  imdb_train,
#  'n_estimators',
#  [50,100,150,200],
#  file_name="AdaBoostIMDB",
#  print_debug=True
#  )


params = {
 'n_estimators': [200] #Best value
}
clf_1 = train_imdb_classifier(AdaBoostClassifier(), print_debug=True, params=params)
print(clf_1.cv_results_)
 # 'mean_test_score': array([0.84066823])
clf_2 = train_newsgroup_classifier(AdaBoostClassifier(), print_debug=True, params=params)
print(clf_2.cv_results_)
# 'mean_test_score': array([0.4060969])

# Find the final accuracies of each classifier
accuracy_imdb = find_classifier_accuracy(clf_1, imdb_test)
accuracy_newsgroup = find_classifier_accuracy(clf_2, newsgroup_test)

print(f"Final accuracy for IMDB: {accuracy_imdb}")
# Final accuracy for IMDB: 0.849804443369219
print(f"Final accuracy for Newsgroup: {accuracy_newsgroup}").
# Final accuracy for Newsgroup: 0.4523187459177008