from ucimlrepo import fetch_ucirepo
import pandas as pd


# fetch dataset
adult = fetch_ucirepo(id=2)

# load data into pandas dataframe
df = pd.DataFrame(data=adult.data.features, columns=adult.feature_names)

# remove missing values
df = df.dropna()

# remove duplicate rows
df = df.drop_duplicates()

# remove rows with invalid values
df = df[(df != '?').all(axis=1)]
