import numpy as np

# fetch dataset
spambase_data = np.loadtxt("../data/spambase/spambase.data", delimiter=',')
arr = np.array(spambase_data)

# remove duplicate rows
arr = np.unique(arr, axis=0)

# normalize capital_run_length_average, capital_run_length_longest, capital_run_length_total columns
arr[:, 54] = (arr[:, 54] - np.mean(arr[:, 54])) / np.std(arr[:, 54])
arr[:, 55] = (arr[:, 55] - np.mean(arr[:, 55])) / np.std(arr[:, 55])
arr[:, 56] = (arr[:, 56] - np.mean(arr[:, 56])) / np.std(arr[:, 56])

# save cleaned dataset
# with open("spambase_data_clean.csv", "w") as f:
#     np.savetxt(f, arr, delimiter=",", fmt="%s")
