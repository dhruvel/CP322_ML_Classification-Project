from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import os

from data_utils import split_data

"""
newsgroup_train: training data from 20newsgroups
newsgroup_test: testing data from 20newsgroups
newsgroup_all: all data from 20newsgroups
"""

# Make sure we are running from Project2/src
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Preparation Done:
# - Removed headers, footers, and quotes
# - Removed duplicates

if not os.path.exists('../data/20news-bydate_py3.pkz') and not os.path.exists('../data/20news-bydate_py.pkz'):
    print("Downloading 20newsgroups data... this will only happen once.")

# Load train and test data, removing the headers, footers, and quotes
newsgroup_all = fetch_20newsgroups(
    data_home='../data/',
    subset='all',
    remove=('headers', 'footers', 'quotes'),
    shuffle=True,
    random_state=0,
)
# Load as a DataFrame and Remove duplicates
newsgroup_all = pd.DataFrame({
    'data': newsgroup_all.data,
    'target': newsgroup_all.target
}).drop_duplicates()

newsgroup_train, newsgroup_test = split_data(newsgroup_all)
print("Loaded 20newsgroups data (train: {}, test: {})".format(len(newsgroup_train), len(newsgroup_test)))
