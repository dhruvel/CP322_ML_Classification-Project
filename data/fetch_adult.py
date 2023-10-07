import numpy as np


# fetch dataset
adult_data = np.loadtxt("CP322_ML_Classification-Project/data/adult/adult.data", delimiter=',', dtype='str')
arr = np.array(adult_data)

# remove rows with missing values
arr = arr[~np.any(arr == " ?", axis=1)]

# remove duplicate rows
arr = np.unique(arr, axis=0)


