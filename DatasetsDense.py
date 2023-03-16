from Tfidf import y_val, x_val_vec_negations, x_val_vec_clean, y_train_clean
from Sampling import *
import pickle
import os
import scipy

# del adasyn_clean, adasyn_negations, ncr_clean, ncr_negations, le_y, ncr_negations, nm_negations, nm_clean
datasets = [
    (X_train_adasyn_negations, y_train_adasyn_negations, x_val_vec_negations, y_val, "ADASYN Negations"),
    (X_train_adasyn_clean, y_train_adasyn_clean, x_val_vec_clean, y_val, "ADASYN Clean"),
    (x_train_vec_negations, y_train_clean, x_val_vec_negations, y_val, "Original Negations"),
    (x_train_vec_clean, y_train_clean, x_val_vec_clean, y_val, "Original Clean"),
    (X_train_ncr_negations, y_train_ncr_negations, x_val_vec_negations, y_val, "NCR Negations"),
    (X_train_ncr_clean, y_train_ncr_clean, x_val_vec_clean, y_val, "NCR Clean"),
    (X_train_nm_negations, y_train_nm_negations, x_val_vec_negations, y_val, "NearMiss Negations"),
    (X_train_nm_clean, y_train_nm_clean, x_val_vec_clean, y_val, "NearMiss Clean")
    ]

for i, (x_train, y_train, x_val, y_val, name) in enumerate(datasets):
    datasets[i] = (x_train.todense(), y_train, x_val.todense(), y_val, name)


# Check if each dataset exists, load from file if it does, save and load if it doesn't
for i, dataset in enumerate(datasets):
    dataset_name = dataset[4]
    file_name = f"Datasets/{dataset_name}_dense.pkl"
    if not os.path.exists(file_name):
        print(f"Saving {dataset_name} Sampled Object...")
        with open(file_name, 'wb') as f:
            pickle.dump(dataset, f)