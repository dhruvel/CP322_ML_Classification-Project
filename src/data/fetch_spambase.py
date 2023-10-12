import numpy as np


# fetch dataset
spambase_data = np.loadtxt("CP322_ML_Classification-Project/data/spambase/spambase.data", delimiter=',')
arr = np.array(spambase_data)

# remove duplicate rows
arr = np.unique(arr, axis=0)

# normalize data
arr = arr / arr.max(axis=0)

# save cleaned dataset
# with open("spambase_cleaned.data", "w") as f:
#     np.savetxt(f, arr, delimiter=',', fmt='%f')

