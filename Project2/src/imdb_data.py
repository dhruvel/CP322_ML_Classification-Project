from sklearn.datasets import load_files
import pandas as pd
import os

"""
- imdb_train: training data
- imdb_test: testing data
- imdb_all: all data
"""

# Make sure we are running from Project2/src
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Preparation Done:
# - Removed reviews without labels
# - Removed duplicate reviews

def _save_to_csv():
    # First download IMDB data from https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    # and extract to Project2/data/aclImdb

    # Load train and test data
    imdb_train = load_files(
        container_path='../data/aclImdb/train',
        categories=['pos', 'neg'],
        shuffle=True,
        random_state=0,
        encoding='ISO-8859-1',
    )

    # Load as a DataFrame and Remove duplicates
    imdb_train = pd.DataFrame({
        'data': imdb_train.data,
        'target': imdb_train.target
    }).drop_duplicates()

    imdb_test = load_files(
        container_path='../data/aclImdb/test',
        categories=['pos', 'neg'],
        shuffle=True,
        random_state=0,
        encoding='ISO-8859-1',
    )

    # Load as a DataFrame and Remove duplicates
    imdb_test = pd.DataFrame({
        'data': imdb_test.data,
        'target': imdb_test.target
    }).drop_duplicates()

    # Save to CSV file
    imdb_train.to_csv('../data/aclImdb/train.csv', index=False)
    imdb_test.to_csv('../data/aclImdb/test.csv', index=False)

if not os.path.exists('../data/aclImdb/train.csv') or not os.path.exists('../data/aclImdb/test.csv'):
    print("Creating IMDB CSVs... this will only happen once.")
    _save_to_csv()

imdb_train = pd.read_csv('../data/aclImdb/train.csv')
imdb_test = pd.read_csv('../data/aclImdb/test.csv')
# Combine the data
imdb_all = pd.concat([imdb_train, imdb_test])

print("Loaded IMDB data (train: {}, test: {})".format(len(imdb_train), len(imdb_test)))
