import numpy as np

# fetch dataset
diabetes_data = np.loadtxt("CP322_ML_Classification-Project/data/diabetes/diabetic_data.csv", delimiter=',', dtype='str')
arr = np.array(diabetes_data)

# remove rows with missing values
arr = arr[~np.any(arr == "?", axis=1)]

# remove duplicate rows
_, idx = np.unique(arr, axis=0, return_index=True)
arr = arr[np.sort(idx)]


# with open("diabetic_data_clean.csv", "w") as f:
#     np.savetxt(f, arr, delimiter=",", fmt="%s")