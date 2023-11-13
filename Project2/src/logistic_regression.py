import utils
from sklearn.linear_model import LogisticRegression
from newsgroup_data import newsgroup_train,newsgroup_test
from imdb_data import imdb_test,imdb_train
from plot import plot_accuracy_over_param
from utils import train_imdb_classifier


# # Regression over different regulization lambdas
plot_accuracy_over_param(LogisticRegression(),
                         newsgroup_train,
                         'C',
                         [1.0,2.0,3.0,4.0],
                         file_name="LogisticRegressionNewsgroup",
                         print_debug=True
                         )

plot_accuracy_over_param(LogisticRegression(),
                         imdb_train,
                         'C',
                         [1.0,2.0,3.0,4.0],
                         file_name="LogisticRegressionIMDB",
                         print_debug=True
                         )

# params = {
#         'C': [4]  
#     }
# clf = train_imdb_classifier(LogisticRegression(), print_debug=True, params=params)
# best_C = clf.best_params_['clf__C']
# print(f"Best C value: {best_C}")
# print(clf.cv_results_)
# #  'mean_test_score': array([0.89475565])
