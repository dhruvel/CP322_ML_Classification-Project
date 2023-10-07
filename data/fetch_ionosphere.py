from ucimlrepo import fetch_ucirepo
import pandas as pd


# fetch dataset
ionosphere = fetch_ucirepo(id=52)

# load data into pandas dataframe
df = pd.DataFrame(data=ionosphere.data.features, columns=ionosphere.feature_names)

# remove missing values
df = df.dropna()

# remove duplicate rows
df = df.drop_duplicates()

# remove rows with invalid values
df = df[(df != '?').all(axis=1)]
