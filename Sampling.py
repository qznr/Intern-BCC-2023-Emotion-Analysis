from tqdm import tqdm
from Tfidf import x_train_vec_negations, x_train_vec_clean, y_train_clean
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import NearMiss

# ADASYN
adasyn_negations = ADASYN(random_state=42)
print("ADASYN Negations sampling")
for i in tqdm(range(1)):
    X_train_adasyn_negations, y_train_adasyn_negations = adasyn_negations.fit_resample(x_train_vec_negations, y_train_clean)

adasyn_clean = ADASYN(random_state=42)
print("ADASYN Clean sampling")
for i in tqdm(range(1)):
    X_train_adasyn_clean, y_train_adasyn_clean = adasyn_clean.fit_resample(x_train_vec_clean, y_train_clean)
# NCR
le_y = LabelEncoder()
y_train_clean_encoded = le_y.fit_transform(y_train_clean)
ncr_negations = NeighbourhoodCleaningRule(n_jobs=-1)
print("NCR Negations sampling")
for i in tqdm(range(1)):
    X_train_ncr_negations, y_train_ncr_negations = ncr_negations.fit_resample(x_train_vec_negations, y_train_clean_encoded)
    y_train_ncr_negations = le_y.inverse_transform(y_train_ncr_negations)

le_y = LabelEncoder()
y_train_clean_encoded = le_y.fit_transform(y_train_clean)
ncr_clean = NeighbourhoodCleaningRule(n_jobs=-1)
print("NCR Clean sampling")
for i in tqdm(range(1)):
    X_train_ncr_clean, y_train_ncr_clean = ncr_clean.fit_resample(x_train_vec_clean, y_train_clean_encoded)
    y_train_ncr_clean = le_y.inverse_transform(y_train_ncr_clean)
# Nearmiss
nm_negations = NearMiss(n_jobs=-1)
print("Nearmiss Negations sampling")
for i in tqdm(range(1)):
    X_train_nm_negations, y_train_nm_negations = nm_negations.fit_resample(x_train_vec_negations, y_train_clean)

nm_clean = NearMiss(n_jobs=-1)
print("Nearmiss Clean sampling")
for i in tqdm(range(1)):
    X_train_nm_clean, y_train_nm_clean = nm_clean.fit_resample(x_train_vec_clean, y_train_clean)