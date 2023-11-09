from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import os

# Make sure we are running from Project2/src
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Preparation Done:
# - Removed headers, footers, and quotes
# - Removed duplicates

if not os.path.exists('../data/20news-bydate_py3.pkz') and not os.path.exists('../data/20news-bydate_py.pkz'):
    print("Downloading 20newsgroups data... this will only happen once.")

# Load train and test data, removing the headers, footers, and quotes
newsgroup_data = fetch_20newsgroups(
    data_home='../data/',
    subset='all',
    remove=('headers', 'footers', 'quotes'),
    shuffle=True,
    random_state=0,
)
# Load as a DataFrame and Remove duplicates
newsgroup_data = pd.DataFrame({
    'data': newsgroup_data.data,
    'target': newsgroup_data.target
}).drop_duplicates()

"""
Splits the IMDB data into train and test dataframes given a ratio.

Parameters:
    ratio: The ratio of train data to test data

Example Usage:
    # Split into 60% train and 40% test
    imdb_train_split, imdb_test_split = split_imdb_data(0.6)
"""
def split_newsgroup_data(ratio=0.5) -> (pd.DataFrame, pd.DataFrame):
    # Split by ratio
    newsgroup_train_split = newsgroup_data[:int(len(newsgroup_data) * ratio)]
    newsgroup_test_split = newsgroup_data[int(len(newsgroup_data) * ratio):]
    return newsgroup_train_split, newsgroup_test_split

newsgroup_train, newsgroup_test = split_newsgroup_data()
print("Loaded 20newsgroups data (train: {}, test: {})".format(len(newsgroup_train), len(newsgroup_test)))
