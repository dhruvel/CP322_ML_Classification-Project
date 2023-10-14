import numpy as np

# fetch dataset
diabetes_data = np.loadtxt("CP322_ML_Classification-Project/data/diabetes/diabetic_data.csv", delimiter=',', dtype='str')
arr = np.array(diabetes_data)

# delete first row
arr = np.delete(arr, 0, axis=0)

# delete rows with too many missing values
arr = arr[np.sum(arr == "?", axis=1) <= 5]

# remove rows with missing values
arr = arr[~np.any(arr == "?", axis=1)]

# remove Column ID
arr = np.delete(arr, [0, 1, 6, 7, 8, 19], axis=1)

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

# encode weight column
weight_classes = ["[0-25)", 
                  "[25-50)", 
                  "[50-75)", 
                  "[75-100)", 
                  "[100-125)", 
                  "[125-150)", 
                  "[150-175)", 
                  "[175-200)", 
                  ">200"]
arr[:, 3] = np.array([weight_classes.index(x) for x in arr[:, 3]])

# encode payer_code column
payer_code_classes = ["MC", 
                      "MD", 
                      "HM", 
                      "UN", 
                      "BC", 
                      "SP", 
                      "CP", 
                      "SI", 
                      "DM", 
                      "CM", 
                      "CH", 
                      "PO", 
                      "WC", 
                      "OT"]
arr[:, 5] = np.array([payer_code_classes.index(x) for x in arr[:, 5]])

# encode medical_specialty column
medical_specialty_classes = ["Pediatrics-Endocrinology",
                            "InternalMedicine",
                            "Family/GeneralPractice",
                            "Cardiology",
                            "Surgery-General",
                            "Orthopedics",
                            "Gastroenterology",
                            "Surgery-Cardiovascular/Thoracic",
                            "Nephrology",
                            "Orthopedics-Reconstructive",
                            "Psychiatry",
                            "Emergency/Trauma",
                            "Pulmonology",
                            "Surgery-Neuro",
                            "Obsterics&Gynecology-GynecologicOnco",
                            "ObstetricsandGynecology",
                            "Pediatrics",
                            "Hematology/Oncology",
                            "Otolaryngology",
                            "Surgery-Colon&Rectal",
                            "Pediatrics-CriticalCare",
                            "Endocrinology",
                            "Urology",
                            "Psychiatry-Child/Adolescent",
                            "Pediatrics-Pulmonology",
                            "Neurology",
                            "Anesthesiology-Pediatric",
                            "Radiology",
                            "Pediatrics-Hematology-Oncology",
                            "Psychology",
                            "Podiatry",
                            "Gynecology",
                            "Oncology",
                            "Pediatrics-Neurology",
                            "Surgery-Plastic",
                            "Surgery-Thoracic",
                            "Surgery-PlasticwithinHeadandNeck",
                            "Ophthalmology",
                            "Surgery-Pediatric",
                            "Pediatrics-EmergencyMedicine",
                            "PhysicalMedicineandRehabilitation",
                            "InfectiousDiseases",
                            "Anesthesiology",
                            "Rheumatology",
                            "AllergyandImmunology",
                            "Surgery-Maxillofacial",
                            "Pediatrics-InfectiousDiseases",
                            "Pediatrics-AllergyandImmunology",
                            "Dentistry",
                            "Surgeon",
                            "Surgery-Vascular",
                            "Osteopath",
                            "Psychiatry-Addictive",
                            "Surgery-Cardiovascular",
                            "PhysicianNotFound",
                            "Hematology",
                            "Proctology",
                            "Obstetrics",
                            "SurgicalSpecialty",
                            "Radiologist",
                            "Pathology",
                            "Dermatology",
                            "SportsMedicine",
                            "Speech",
                            "Hospitalist",
                            "OutreachServices",
                            "Cardiology-Pediatric",
                            "Perinatology",
                            "Neurophysiology",
                            "Endocrinology-Metabolism",
                            "DCPTEAM",
                            "Resident"
]
arr[:, 6] = np.array([medical_specialty_classes.index(x) for x in arr[:, 6]])

# encode max_glu_serum column
max_glu_serum_classes = ["None"]
arr[:, 16] = np.array([max_glu_serum_classes.index(x) for x in arr[:, 16]])

# encode A1Cresult column
A1Cresult_classes = ["None", 
                     ">7", 
                     ">8", 
                     "Norm"]
arr[:, 17] = np.array([A1Cresult_classes.index(x) for x in arr[:, 17]])

# encode metformin column
metformin_classes = ["No", 
                     "Steady", 
                     "Up", 
                     "Down"]
arr[:, 18] = np.array([metformin_classes.index(x) for x in arr[:, 18]])

# encode repaglinide column
repaglinide_classes = ["No", 
                       "Steady", 
                       "Up", 
                       "Down"]
arr[:, 19] = np.array([repaglinide_classes.index(x) for x in arr[:, 19]])

# encode nateglinide column
nateglinide_classes = ["No", 
                       "Steady", 
                       "Up", 
                       "Down"]
arr[:, 20] = np.array([nateglinide_classes.index(x) for x in arr[:, 20]])

# encode chlorpropamide column
chlorpropamide_classes = ["No", 
                          "Steady", 
                          "Up", 
                          "Down"]
arr[:, 21] = np.array([chlorpropamide_classes.index(x) for x in arr[:, 21]])

# encode glimepiride column
glimepiride_classes = ["No", 
                       "Steady", 
                       "Up", 
                       "Down"]
arr[:, 22] = np.array([glimepiride_classes.index(x) for x in arr[:, 22]])

# encode acetohexamide column
acetohexamide_classes = ["No", 
                         "Steady", 
                         "Up", 
                         "Down"]
arr[:, 23] = np.array([acetohexamide_classes.index(x) for x in arr[:, 23]])

# encode glipizide column
glipizide_classes = ["No", 
                     "Steady", 
                     "Up", 
                     "Down"]
arr[:, 24] = np.array([glipizide_classes.index(x) for x in arr[:, 24]])

# encode glyburide column
glyburide_classes = ["No", 
                     "Steady", 
                     "Up", 
                     "Down"]
arr[:, 25] = np.array([glyburide_classes.index(x) for x in arr[:, 25]])

# encode tolbutamide column
tolbutamide_classes = ["No", 
                       "Steady", 
                       "Up", 
                       "Down"]
arr[:, 26] = np.array([tolbutamide_classes.index(x) for x in arr[:, 26]])

# encode pioglitazone column
pioglitazone_classes = ["No", 
                        "Steady", 
                        "Up", 
                        "Down"]
arr[:, 27] = np.array([pioglitazone_classes.index(x) for x in arr[:, 27]])

# encode rosiglitazone column
rosiglitazone_classes = ["No", 
                         "Steady", 
                         "Up", 
                         "Down"]
arr[:, 28] = np.array([rosiglitazone_classes.index(x) for x in arr[:, 28]])

# encode acarbose column
acarbose_classes = ["No", 
                    "Steady", 
                    "Up", 
                    "Down"]
arr[:, 29] = np.array([acarbose_classes.index(x) for x in arr[:, 29]])

# encode miglitol column
miglitol_classes = ["No", 
                    "Steady", 
                    "Up", 
                    "Down"]
arr[:, 30] = np.array([miglitol_classes.index(x) for x in arr[:, 30]])

# encode troglitazone column
troglitazone_classes = ["No", 
                        "Steady", 
                        "Up", 
                        "Down"]
arr[:, 31] = np.array([troglitazone_classes.index(x) for x in arr[:, 31]])

# encode tolazamide column
tolazamide_classes = ["No", 
                      "Steady", 
                      "Up", 
                      "Down"]
arr[:, 32] = np.array([tolazamide_classes.index(x) for x in arr[:, 32]])

# encode examide column
examide_classes = ["No", 
                   "Steady", 
                   "Up", 
                   "Down"]
arr[:, 33] = np.array([examide_classes.index(x) for x in arr[:, 33]])

# encode citoglipton column
citoglipton_classes = ["No", 
                       "Steady", 
                       "Up", 
                       "Down"]
arr[:, 34] = np.array([citoglipton_classes.index(x) for x in arr[:, 34]])

# encode insulin column
insulin_classes = ["No", 
                   "Steady", 
                   "Up", 
                   "Down"]
arr[:, 35] = np.array([insulin_classes.index(x) for x in arr[:, 35]])

# encode glyburide-metformin column
glyburide_metformin_classes = ["No", 
                               "Steady", 
                               "Up", 
                               "Down"]
arr[:, 36] = np.array([glyburide_metformin_classes.index(x) for x in arr[:, 36]])

# encode glipizide-metformin column
glipizide_metformin_classes = ["No", 
                               "Steady", 
                               "Up", 
                               "Down"]
arr[:, 37] = np.array([glipizide_metformin_classes.index(x) for x in arr[:, 37]])

# encode glimepiride-pioglitazone column
glimepiride_pioglitazone_classes = ["No", 
                                    "Steady", 
                                    "Up", 
                                    "Down"]
arr[:, 38] = np.array([glimepiride_pioglitazone_classes.index(x) for x in arr[:, 38]])

# encode metformin-rosiglitazone column
metformin_rosiglitazone_classes = ["No", 
                                   "Steady", 
                                   "Up", 
                                   "Down"]
arr[:, 39] = np.array([metformin_rosiglitazone_classes.index(x) for x in arr[:, 39]])

# encode metformin-pioglitazone column
metformin_pioglitazone_classes = ["No", 
                                  "Steady", 
                                  "Up", 
                                  "Down"]
arr[:, 40] = np.array([metformin_pioglitazone_classes.index(x) for x in arr[:, 40]])

# encode change column
change_classes = ["No", 
                  "Ch"]
arr[:, 41] = np.array([change_classes.index(x) for x in arr[:, 41]])

# encode diabetesMed column
diabetesMed_classes = ["No", 
                       "Yes"]
arr[:, 42] = np.array([diabetesMed_classes.index(x) for x in arr[:, 42]])

# encode readmitted column
readmitted_classes = ["NO", 
                      ">30", 
                      "<30"]
arr[:, 43] = np.array([readmitted_classes.index(x) for x in arr[:, 43]])


# remove duplicate rows
_, idx = np.unique(arr, axis=0, return_index=True)
arr = arr[np.sort(idx)]

# with open("diabetic_data_clean.csv", "w") as f:
#     np.savetxt(f, arr, delimiter=",", fmt="%s")