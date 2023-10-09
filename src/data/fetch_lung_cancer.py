import numpy as np

# fetch dataset
lung_cancer_data = np.loadtxt("../data/lung_cancer/lung_cancer.data", delimiter=',', dtype='str')
arr = np.array(lung_cancer_data)

# remove rows with missing values
arr = arr[~np.any(arr == " ?", axis=1)]

# remove duplicate rows
arr = np.unique(arr, axis=0)
