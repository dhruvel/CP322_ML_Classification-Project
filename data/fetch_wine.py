from ucimlrepo import fetch_ucirepo
import pandas as pd


# fetch dataset
wine = fetch_ucirepo(id=109)

# load data into pandas dataframe
df = pd.DataFrame(data=wine.data.features, columns=wine.feature_names)

# remove missing values
df = df.dropna()

# remove duplicate rows
df = df.drop_duplicates()

# remove rows with invalid values
df = df[(df != '?').all(axis=1)]
