import numpy as np


# fetch dataset
adult_data = np.loadtxt("../data/adult/adult.data", delimiter=',', dtype='str')
arr = np.array(adult_data)

# remove rows with missing values
arr = arr[~np.any(arr == " ?", axis=1)]

# remove duplicate rows
arr = np.unique(arr, axis=0)


