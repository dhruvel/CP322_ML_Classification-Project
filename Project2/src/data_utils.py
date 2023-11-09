"""
Splits data into train and test dataframes given a ratio.

Parameters:
    data: The data to split
    ratio: The ratio of train data to test data

Example Usage:
    # Split into 60% train and 40% test
    from imdb_data import imdb_all
    imdb_train, imdb_test = split_data(imdb_all, ratio=0.6)
"""
def split_data(data, ratio=0.5):
    # Shuffle data
    data = data.sample(frac=1)
    # Split by ratio
    train = data[:int(len(data) * ratio)]
    test = data[int(len(data) * ratio):]

    return train, test