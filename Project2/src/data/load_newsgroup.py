from sklearn.datasets import fetch_20newsgroups
import pandas as pd

# Load train and test data, removing the headers, footers, and quotes
newsgroup_train = fetch_20newsgroups(
    data_home='../../data/',
    subset='train',
    remove=('headers', 'footers', 'quotes'),
    shuffle=True,
    random_state=0,
)
# Load as a DataFrame and Remove duplicates
newsgroup_train = pd.DataFrame({
    'data': newsgroup_train.data,
    'target': newsgroup_train.target
}).drop_duplicates()

newsgroup_test = fetch_20newsgroups(
    data_home='../../data/',
    subset='test',
    remove=('headers', 'footers', 'quotes'),
    shuffle=True,
    random_state=0,
)
newsgroup_test = pd.DataFrame({
    'data': newsgroup_test.data,
    'target': newsgroup_test.target
}).drop_duplicates()