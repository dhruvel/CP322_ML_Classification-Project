import numpy as np

# fetch dataset
diabetes_data = np.loadtxt("CP322_ML_Classification-Project/data/diabetes/diabetic_data.csv", delimiter=',', dtype='str')
arr = np.array(diabetes_data)

# delete first row
arr = np.delete(arr, 0, axis=0)

# remove rows with missing values
arr = arr[~np.any(arr == "?", axis=1)]

# remove ID
arr = np.delete(arr, [0,1], axis=1)

# encode race column
race_classes = ["Caucasian", 
                "Asian", 
                "AfricanAmerican", 
                "Hispanic", 
                "Other"]
arr[:, 0] = np.array([race_classes.index(x) for x in arr[:, 0]])

# encode gender column
gender_classes = ["Male", 
                  "Female"]
arr[:, 1] = np.array([gender_classes.index(x) for x in arr[:, 1]])

# remove duplicate rows
_, idx = np.unique(arr, axis=0, return_index=True)
arr = arr[np.sort(idx)]

# with open("diabetic_data_clean.csv", "w") as f:
#     np.savetxt(f, arr, delimiter=",", fmt="%s")