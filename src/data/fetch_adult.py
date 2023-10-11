import numpy as np


# fetch dataset
adult_data = np.loadtxt("../data/adult/adult.data", delimiter=',', dtype='str')
arr = np.array(adult_data)

# remove rows with missing values
arr = arr[~np.any(arr == " ?", axis=1)]
# remove duplicate rows
arr = np.unique(arr, axis=0)
# trim whitespace
arr = np.char.strip(arr)

# Turn categorical data into ordinal data
# workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked
workclass_classes = [
    "Private",
    "Self-emp-not-inc",
    "Self-emp-inc",
    "Federal-gov",
    "Local-gov",
    "State-gov",
    "Without-pay",
    "Never-worked",
]
arr[:, 1] = np.array([workclass_classes.index(x) for x in arr[:, 1]])

# education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education_classes = [
    "Bachelors",
    "Some-college",
    "11th",
    "HS-grad",
    "Prof-school",
    "Assoc-acdm",
    "Assoc-voc",
    "9th",
    "7th-8th",
    "12th",
    "Masters",
    "1st-4th",
    "10th",
    "Doctorate",
    "5th-6th",
    "Preschool",
]
arr[:, 3] = np.array([education_classes.index(x) for x in arr[:, 3]])

# marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
marital_status_classes = [
    "Married-civ-spouse",
    "Divorced",
    "Never-married",
    "Separated",
    "Widowed",
    "Married-spouse-absent",
    "Married-AF-spouse",
]
arr[:, 5] = np.array([marital_status_classes.index(x) for x in arr[:, 5]])

# occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
occupation_classes = [
    "Tech-support",
    "Craft-repair",
    "Other-service",
    "Sales",
    "Exec-managerial",
    "Prof-specialty",
    "Handlers-cleaners",
    "Machine-op-inspct",
    "Adm-clerical",
    "Farming-fishing",
    "Transport-moving",
    "Priv-house-serv",
    "Protective-serv",
    "Armed-Forces",
]
arr[:, 6] = np.array([occupation_classes.index(x) for x in arr[:, 6]])

# relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
relationship_classes = [
    "Wife",
    "Own-child",
    "Husband",
    "Not-in-family",
    "Other-relative",
    "Unmarried",
]
arr[:, 7] = np.array([relationship_classes.index(x) for x in arr[:, 7]])

# race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
race_classes = [
    "White",
    "Asian-Pac-Islander",
    "Amer-Indian-Eskimo",
    "Other",
    "Black",
]
arr[:, 8] = np.array([race_classes.index(x) for x in arr[:, 8]])

# sex: Female, Male.
sex_classes = [
    "Female",
    "Male",
]
arr[:, 9] = np.array([sex_classes.index(x) for x in arr[:, 9]])

# native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
native_country_classes = [
    "United-States",
    "Cambodia",
    "England",
    "Puerto-Rico",
    "Canada",
    "Germany",
    "Outlying-US(Guam-USVI-etc)",
    "India",
    "Japan",
    "Greece",
    "South",
    "China",
    "Cuba",
    "Iran",
    "Honduras",
    "Philippines",
    "Italy",
    "Poland",
    "Jamaica",
    "Vietnam",
    "Mexico",
    "Portugal",
    "Ireland",
    "France",
    "Dominican-Republic",
    "Laos",
    "Ecuador",
    "Taiwan",
    "Haiti",
    "Columbia",
    "Hungary",
    "Guatemala",
    "Nicaragua",
    "Scotland",
    "Thailand",
    "Yugoslavia",
    "El-Salvador",
    "Trinadad&Tobago",
    "Peru",
    "Hong",
    "Holand-Netherlands",
]
arr[:, 13] = np.array([native_country_classes.index(x) for x in arr[:, 13]])

# convert labels to binary
labels = arr[:, -1]
labels[labels == "<=50K"] = 0
labels[labels == ">50K"] = 1
arr[:, -1] = labels

# make all values floats
arr = arr.astype(np.float)

# normalize fnlwgt, capital-gain, capital-loss
arr[:, 2] = (arr[:, 2] - np.mean(arr[:, 2])) / np.std(arr[:, 2])
arr[:, 10] = (arr[:, 10] - np.mean(arr[:, 10])) / np.std(arr[:, 10])
arr[:, 11] = (arr[:, 11] - np.mean(arr[:, 11])) / np.std(arr[:, 11])