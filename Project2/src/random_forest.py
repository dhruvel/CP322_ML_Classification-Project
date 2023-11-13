import utils
from sklearn.ensemble import RandomForestClassifier
from newsgroup_data import newsgroup_train,newsgroup_test
from imdb_data import imdb_test,imdb_train
from plot import plot_accuracy_over_param
from utils import train_imdb_classifier, train_newsgroup_classifier


# Testing random foreest with different methods of determining split

#IMDB
# resultsImdb = []
# for i in ['gini','entropy','log_loss']:
#     accuracies = plot_accuracy_over_param(RandomForestClassifier(),
#                             imdb_train,
#                             'criterion',
#                             [i],
#                             file_name=(f"RandomForest{i}"),
#                             print_debug=True
#                             )
#     resultsImdb.append(accuracies)
# print(resultsImdb)

# # results: [array([0.83480576]), array([0.83605056]), array([0.83845977])]
    
# #Newsgroup
# resultsImdb = []
# for i in ['gini','entropy','log_loss']:
#     plot_accuracy_over_param(RandomForestClassifier(),
#                             newsgroup_train,
#                             'criterion',
#                             [i],
#                             file_name=(f"RandomForest{i}"),
#                             print_debug=True
#                             )
#     resultsImdb.append(accuracies)
# print(resultsImdb)

# # Results: [array([0.83845977]), array([0.83845977]), array([0.83845977])]

# Test with minSamples per leaf
# plot_accuracy_over_param(RandomForestClassifier(),
#                             newsgroup_train,
#                             'min_samples_leaf',
#                             [1,2,4,6,8],
#                             file_name=(f"RandomForestMinSamplesLeafNewsgroup"),
#                             print_debug=True
#                             )

# plot_accuracy_over_param(RandomForestClassifier(),
#                             imdb_train,
#                             'min_samples_leaf',
#                             [1,2,4,6,8],
#                             file_name=(f"RandomForestMinSamplesLeafImdb"),
#                             print_debug=True
#                             )

# # Min samples split
# plot_accuracy_over_param(RandomForestClassifier(),
#                         newsgroup_train,
#                         'min_samples_split',
#                         [1,2,4,6,8],
#                         file_name=(f"RandomForestMinSamplesSplitNewsgroup"),
#                         print_debug=True
#                         )

# plot_accuracy_over_param(RandomForestClassifier(),
#                         newsgroup_train,
#                         'min_samples_split',
#                         [1,2,4,6,8],
#                         file_name=(f"RandomForestMinSamplesSplitImdb"),
#                         print_debug=True
#                         )

# # # Number of trees generated
# plot_accuracy_over_param(RandomForestClassifier(),
#                         newsgroup_train,
#                         'n_estimators',
#                         [70,80,90,100,110,120,125,130,135,140],
#                         file_name=(f"RandomForestNumOfTreesNewsgroup"),
#                         print_debug=True
#                         )

# plot_accuracy_over_param(RandomForestClassifier(),
#                         newsgroup_train,
#                         'n_estimators',
#                         [70,80,90,100,110,120,125,130,135,140],
#                         file_name=(f"RandomForestNumOfTreesImdb"),
#                         print_debug=True
#                         )

# # Max subset of features for each tree
# plot_accuracy_over_param(RandomForestClassifier(),
#                         newsgroup_train,
#                         'max_features',
#                         ['sqrt', 'log2'],
#                         file_name=(f"RandomForestMaxFeaturesNewsgroup"),
#                         print_debug=True
#                         )

# plot_accuracy_over_param(RandomForestClassifier(),
#                         newsgroup_train,
#                         'max_features',
#                         ['sqrt', 'log2'],
#                         file_name=(f"RandomForestMaxFeaturesImdb"),
#                         print_debug=True
#                         )]

# validating with optimal Parameters
params = {
        'n_estimators': [130],
        'min_samples_split': [2],
        'min_samples_leaf':[2],
        'criterion':['gini'],
        'max_features':['sqrt']
    }
# clf = train_imdb_classifier(RandomForestClassifier(), print_debug=True, params=params)
# print(clf.cv_results_)
# 'mean_test_score': array([0.84082891])
clf = train_newsgroup_classifier(RandomForestClassifier(), print_debug=True, params=params)
print(clf.cv_results_)
# 'mean_test_score': array([0.6547632])