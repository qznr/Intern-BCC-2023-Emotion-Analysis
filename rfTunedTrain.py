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

def objective(trial, x_train, y_train, x_val, y_val):
    rf_n_estimators = trial.suggest_int("n_estimators", 10, 1000)
    rf_max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
    rf_min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    rf_min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    rf_model = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, min_samples_split=rf_min_samples_split, min_samples_leaf=rf_min_samples_leaf)
    rf_model.fit(x_train, y_train)
    y_pred = rf_model.predict(x_val)
    precision = precision_score(y_val, y_pred, average='macro')
    return precision

if __name__ == '__main__':
    for x_train, y_train, x_val, y_val, dataset_name in datasets:
        # Create Optuna study for each datasets with MySQL storage