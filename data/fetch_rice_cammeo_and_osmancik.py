from ucimlrepo import fetch_ucirepo
import pandas as pd


# fetch dataset
rice_cammeo_and_osmancik = fetch_ucirepo(id=545)

# load data into pandas dataframe
df = pd.DataFrame(data=rice_cammeo_and_osmancik.data.features, columns=rice_cammeo_and_osmancik.feature_names)

# remove missing values
df = df.dropna()

# remove duplicate rows
df = df.drop_duplicates()

# remove rows with invalid values
df = df[(df != '?').all(axis=1)]
