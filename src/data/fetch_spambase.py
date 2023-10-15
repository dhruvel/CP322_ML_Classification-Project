import numpy as np


# fetch dataset
spambase_data = np.loadtxt("../data/spambase/spambase.data", delimiter=',')
arr = np.array(spambase_data)

# remove duplicate rows
arr = np.unique(arr, axis=0)

""" # save cleaned dataset
with open("spambase_data_clean.csv", "w") as f:
    np.savetxt(f, arr, delimiter=",", fmt="%s")

 """