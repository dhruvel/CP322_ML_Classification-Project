import numpy as np


# fetch dataset
wine_data = np.loadtxt("../data/wine/wine.data", delimiter=',', dtype='str')
arr = np.array(wine_data)

# remove rows with missing values
arr = arr[~np.any(arr == " ?", axis=1)]

# remove duplicate rows
arr = np.unique(arr, axis=0)
