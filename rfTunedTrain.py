import os
from Tfidf import y_test
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

datasets = [
    ('X_train_adasyn_negations', 'y_train_adasyn_negations', 'x_val_vec_negations', 'y_test', "ADASYN Negations"),
    ('X_train_adasyn_clean', 'y_train_adasyn_clean', 'x_val_vec_clean', 'y_test', "ADASYN Clean"),
    ('x_train_vec_negations', 'y_train_clean', 'x_val_vec_negations', 'y_test', "Original Negations"),
    ('x_train_vec_clean', 'y_train_clean', 'x_val_vec_clean', 'y_test', "Original Clean"),
    ('X_train_ncr_negations', 'y_train_ncr_negations', 'x_val_vec_negations', 'y_test', "NCR Negations"),
    ('X_train_ncr_clean', 'y_train_ncr_clean', 'x_val_vec_clean', 'y_test', "NCR Clean"),
    ('X_train_nm_negations', 'y_train_nm_negations', 'x_val_vec_negations', 'y_test', "NearMiss Negations"),
    ('X_train_nm_clean', 'y_train_nm_clean', 'x_val_vec_clean', 'y_test', "NearMiss Clean")
    ]

if __name__ == '__main__':
    # Train for each datasets
    for x_train, y_train, x_val, y_test, dataset_name in datasets:
        