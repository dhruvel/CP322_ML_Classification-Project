import numpy as np

# fetch dataset
lung_cancer_data = np.loadtxt("CP322_ML_Classification-Project/data/lung_cancer/lung_cancer.data", delimiter=',', dtype='str')
arr = np.array(lung_cancer_data)

# remove rows with missing values
arr = arr[~np.any(arr == " ?", axis=1)]

# remove duplicate rows
arr = np.unique(arr, axis=0)

with open("CP322_ML_Classification-Project/data/lung_clean.data", 'w') as f:
    # write contents of array to file
    for row in arr:
        f.write(','.join(row) + '\n')

    f.close()