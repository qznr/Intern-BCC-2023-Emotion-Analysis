from Tfidf import y_val, y_test, x_val_vec_negations, x_val_vec_clean, y_train_clean, x_test_vec_clean, x_test_vec_negations
from Sampling import *
import pickle
import os

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

datasets_test = [
    (X_train_adasyn_negations, y_train_adasyn_negations, x_test_vec_negations, y_test, "ADASYN Negations"),
    (X_train_adasyn_clean, y_train_adasyn_clean, x_test_vec_clean, y_test, "ADASYN Clean"),
    (x_train_vec_negations, y_train_clean, x_test_vec_negations, y_test, "Original Negations"),
    (x_train_vec_clean, y_train_clean, x_test_vec_clean, y_test, "Original Clean"),
    (X_train_ncr_negations, y_train_ncr_negations, x_test_vec_negations, y_test, "NCR Negations"),
    (X_train_ncr_clean, y_train_ncr_clean, x_test_vec_clean, y_test, "NCR Clean"),
    (X_train_nm_negations, y_train_nm_negations, x_test_vec_negations, y_test, "NearMiss Negations"),
    (X_train_nm_clean, y_train_nm_clean, x_test_vec_clean, y_test, "NearMiss Clean")
    ]

if not os.path.exists("Datasets/Test"):
    os.mkdir("Datasets/Test")
if not os.path.exists("Datasets/Val"):
    os.mkdir("Datasets/Val")

# Check if each dataset exists, save if it doesn't
for i, dataset in enumerate(datasets):
    dataset_name = dataset[4]
    file_name = f"Datasets/Val/{dataset_name}.pkl"
    if not os.path.exists(file_name):
        print(f"Saving {dataset_name} Val Sampled Object...")
        with open(file_name, 'wb') as f:
            pickle.dump(dataset, f)
    else :
        print(f"Saved {dataset_name} Val Sampled Object already exist")

# Check if each dataset exists, load from file if it does, save and load if it doesn't
for i, dataset in enumerate(datasets_test):
    dataset_name = dataset[4]
    file_name = f"Datasets/Test/{dataset_name}.pkl"
    if not os.path.exists(file_name):
        print(f"Saving {dataset_name} Test Sampled Object...")
        with open(file_name, 'wb') as f:
            pickle.dump(dataset, f)
    else :
        print(f"Saved {dataset_name} Test Sampled Object already exist")
