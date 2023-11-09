from sklearn.datasets import load_files
import pandas as pd
import os

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
print("Loaded IMDB data (train: {}, test: {})".format(len(imdb_train), len(imdb_test)))

"""
Splits the IMDB data into train and test dataframes given a ratio.

Parameters:
    ratio: The ratio of train data to test data

Example Usage:
    # Split into 60% train and 40% test
    imdb_train_split, imdb_test_split = split_imdb_data(0.6)
"""
def split_imdb_data(ratio=0.5) -> (pd.DataFrame, pd.DataFrame):
    # Add all the data together
    imdb_data = pd.concat([imdb_train, imdb_test])
    # Shuffle
    imdb_data = imdb_data.sample(frac=1)
    # Split by ratio
    imdb_train_split = imdb_data[:int(len(imdb_data) * ratio)]
    imdb_test_split = imdb_data[int(len(imdb_data) * ratio):]

    return imdb_train_split, imdb_test_split
