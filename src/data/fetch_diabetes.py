import math
import re
import numpy as np

# fetch dataset
diabetes_data = np.loadtxt("../data/diabetes/diabetic_data.csv", delimiter=',', dtype='str')
arr = np.array(diabetes_data)

# delete first row
arr = np.delete(arr, 0, axis=0)

# count number of missing values in each column
# for i in range(arr.shape[1]):
#     print(i, np.sum(arr[:, i] == "?"))

# weight (5) column has 98569 missing values, although it would be an interesting feature, it is simply not existent for most patients
# payer_code (10) column has 40256 missing values and is not useful
# medical_specialty (11) column has 49949 missing values but could be an interesting feature

# remove ID columns, weight, and payer_code columns
arr = np.delete(arr, [0, 1, 5, 10, 11], axis=1)

# remove rows with missing values
arr = arr[~np.any(arr == "?", axis=1)]
# with these removed there are 49735 rows left, enough for our purposes

# encode race column
race_classes = ["Caucasian", 
                "Asian", 
                "AfricanAmerican", 
                "Hispanic", 
                "Other"]
arr[:, 0] = np.array([race_classes.index(x) for x in arr[:, 0]])

# remove Unknown/Invalid values
arr = arr[arr[:, 1] != "Unknown/Invalid"]
# encode gender column
gender_classes = ["Male", "Female"]
arr[:, 1] = np.array([gender_classes.index(x) for x in arr[:, 1]])

# encode age column
age_classes = ["[0-10)", 
               "[10-20)", 
               "[20-30)", 
               "[30-40)", 
               "[40-50)", 
               "[50-60)", 
               "[60-70)", 
               "[70-80)", 
               "[80-90)", 
               "[90-100)"]
arr[:, 2] = np.array([age_classes.index(x) for x in arr[:, 2]])

# encode medical_specialty column
# medical_specialty_classes = ["Pediatrics-Endocrinology",
#                             "InternalMedicine",
#                             "Family/GeneralPractice",
#                             "Cardiology",
#                             "Surgery-General",
#                             "Orthopedics",
#                             "Gastroenterology",
#                             "Surgery-Cardiovascular/Thoracic",
#                             "Nephrology",
#                             "Orthopedics-Reconstructive",
#                             "Psychiatry",
#                             "Emergency/Trauma",
#                             "Pulmonology",
#                             "Surgery-Neuro",
#                             "Obsterics&Gynecology-GynecologicOnco",
#                             "ObstetricsandGynecology",
#                             "Pediatrics",
#                             "Hematology/Oncology",
#                             "Otolaryngology",
#                             "Surgery-Colon&Rectal",
#                             "Pediatrics-CriticalCare",
#                             "Endocrinology",
#                             "Urology",
#                             "Psychiatry-Child/Adolescent",
#                             "Pediatrics-Pulmonology",
#                             "Neurology",
#                             "Anesthesiology-Pediatric",
#                             "Radiology",
#                             "Pediatrics-Hematology-Oncology",
#                             "Psychology",
#                             "Podiatry",
#                             "Gynecology",
#                             "Oncology",
#                             "Pediatrics-Neurology",
#                             "Surgery-Plastic",
#                             "Surgery-Thoracic",
#                             "Surgery-PlasticwithinHeadandNeck",
#                             "Ophthalmology",
#                             "Surgery-Pediatric",
#                             "Pediatrics-EmergencyMedicine",
#                             "PhysicalMedicineandRehabilitation",
#                             "InfectiousDiseases",
#                             "Anesthesiology",
#                             "Rheumatology",
#                             "AllergyandImmunology",
#                             "Surgery-Maxillofacial",
#                             "Pediatrics-InfectiousDiseases",
#                             "Pediatrics-AllergyandImmunology",
#                             "Dentistry",
#                             "Surgeon",
#                             "Surgery-Vascular",
#                             "Osteopath",
#                             "Psychiatry-Addictive",
#                             "Surgery-Cardiovascular",
#                             "PhysicianNotFound",
#                             "Hematology",
#                             "Proctology",
#                             "Obstetrics",
#                             "SurgicalSpecialty",
#                             "Radiologist",
#                             "Pathology",
#                             "Dermatology",
#                             "SportsMedicine",
#                             "Speech",
#                             "Hospitalist",
#                             "OutreachServices",
#                             "Cardiology-Pediatric",
#                             "Perinatology",
#                             "Neurophysiology",
#                             "Endocrinology-Metabolism",
#                             "DCPTEAM",
#                             "Resident"
# ]
# arr[:, 7] = np.array([medical_specialty_classes.index(x) for x in arr[:, 7]])

# TODO: There's a lot of classes in these 3 columns, might want to normalize them
# Map ICD-9 codes to disease categories
diag_classes = [
    [1, 139], # Infectious and Parasitic Diseases
    [140, 239], # Neoplasms
    [240, 279], # Endocrine, Nutritional and Metabolic Diseases, and Immunity Disorders
    [280, 289], # Diseases of the Blood and Blood-forming Organs
    [290, 319], # Mental Disorders
    [320, 389], # Diseases of the Nervous System and Sense Organs
    [390, 459], # Diseases of the Circulatory System
    [460, 519], # Diseases of the Respiratory System
    [520, 579], # Diseases of the Digestive System
    [580, 629], # Diseases of the Genitourinary System
    [630, 679], # Complications of Pregnancy, Childbirth, and the Puerperium
    [680, 709], # Diseases of the Skin and Subcutaneous Tissue
    [710, 739], # Diseases of the Musculoskeletal System and Connective Tissue
    [740, 759], # Congenital Anomalies
    [760, 779], # Certain Conditions originating in the Perinatal Period
    [780, 799], # Symptoms, Signs and Ill-defined Conditions
    [800, 999], # Injury and Poisoning
    [1000, 1399], # Supplementary Classification of Factors Influencing Health Status and Contact with Health Services
]

# map diag_1 column containing letters to 1000
arr[:, 13] = np.array([1000 if re.search('[A-Za-z]', x) else x for x in arr[:, 13]])
# map diag_1 column to disease category
arr[:, 13] = np.array([np.where([math.floor(float(x)) >= diag_classes[i][0] and math.floor(float(x)) <= diag_classes[i][1] for i in range(len(diag_classes))])[0][0] for x in arr[:, 13]])

# map diag_2 column containing letters to 1000
arr[:, 14] = np.array([1000 if re.search('[A-Za-z]', x) else x for x in arr[:, 14]])
# map diag_2 column to disease category
arr[:, 14] = np.array([np.where([math.floor(float(x)) >= diag_classes[i][0] and math.floor(float(x)) <= diag_classes[i][1] for i in range(len(diag_classes))])[0][0] for x in arr[:, 14]])

# map diag_3 column containing letters to 1000
arr[:, 15] = np.array([1000 if re.search('[A-Za-z]', x) else x for x in arr[:, 15]])
# map diag_3 column to disease category
arr[:, 15] = np.array([np.where([math.floor(float(x)) >= diag_classes[i][0] and math.floor(float(x)) <= diag_classes[i][1] for i in range(len(diag_classes))])[0][0] for x in arr[:, 15]])

# encode max_glu_serum column
max_glu_serum_classes = ["None", "Norm", ">200", ">300"]
arr[:, 17] = np.array([max_glu_serum_classes.index(x) for x in arr[:, 17]])

# encode A1Cresult column
A1Cresult_classes = ["None", ">8", ">7", "Norm"]
arr[:, 18] = np.array([A1Cresult_classes.index(x) for x in arr[:, 18]])

# encode rest of test and exam columns (columns 19 - 41)
result_classes = ["No", "Up", "Down", "Steady"]
for i in range(19, 42):
    arr[:, i] = np.array([result_classes.index(x) for x in arr[:, i]])

# encode change column
change_classes = ["No", "Ch"]
arr[:, 42] = np.array([change_classes.index(x) for x in arr[:, 42]])

# encode diabetesMed column
diabetesMed_classes = ["No", "Yes"]
arr[:, 43] = np.array([diabetesMed_classes.index(x) for x in arr[:, 43]])

# remove readmitted rows with "NO"
arr = arr[arr[:, 44] != "NO"]
# encode readmitted column
readmitted_classes = [">30", "<30"]
arr[:, 44] = np.array([readmitted_classes.index(x) for x in arr[:, 44]])

# make all values floats
arr = arr.astype(float)

# with open("diabetic_data_clean.csv", "w") as f:
#     np.savetxt(f, arr, delimiter=",", fmt="%s")