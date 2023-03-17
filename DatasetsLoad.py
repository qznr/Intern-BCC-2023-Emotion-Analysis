import pickle
datasets = [
    ('X_train_adasyn_negations', 'y_train_adasyn_negations', 'x_val_vec_negations', 'y_val', "ADASYN Negations"),
    ('X_train_adasyn_clean', 'y_train_adasyn_clean', 'x_val_vec_clean', 'y_val', "ADASYN Clean"),
    ('x_train_vec_negations', 'y_train_clean', 'x_val_vec_negations', 'y_val', "Original Negations"),
    ('x_train_vec_clean', 'y_train_clean', 'x_val_vec_clean', 'y_val', "Original Clean"),
    ('X_train_ncr_negations', 'y_train_ncr_negations', 'x_val_vec_negations', 'y_val', "NCR Negations"),
    ('X_train_ncr_clean', 'y_train_ncr_clean', 'x_val_vec_clean', 'y_val', "NCR Clean"),
    ('X_train_nm_negations', 'y_train_nm_negations', 'x_val_vec_negations', 'y_val', "NearMiss Negations"), # +2500 Trials
    ('X_train_nm_clean', 'y_train_nm_clean', 'x_val_vec_clean', 'y_val', "NearMiss Clean") # +2500 Trials
    ]

datasets_test = [
    # ('X_train_adasyn_negations', 'y_train_adasyn_negations', 'x_test_vec_negations', 'y_test', "ADASYN Negations"),
    # ('X_train_adasyn_clean', 'y_train_adasyn_clean', 'x_test_vec_clean', 'y_test', "ADASYN Clean"),
    # ('x_train_vec_negations', 'y_train_clean', 'x_test_vec_negations', 'y_test', "Original Negations"),
    # ('x_train_vec_clean', 'y_train_clean', 'x_test_vec_clean', 'y_test', "Original Clean"),
    # ('X_train_ncr_negations', 'y_train_ncr_negations', 'x_test_vec_negations', 'y_test', "NCR Negations"),
    # ('X_train_ncr_clean', 'y_train_ncr_clean', 'x_test_vec_clean', 'y_test', "NCR Clean"),
    # ('X_train_nm_negations', 'y_train_nm_negations', 'x_test_vec_negations', 'y_test', "NearMiss Negations"),
    # ('X_train_nm_clean','y_train_nm_clean', 'x_test_vec_clean', 'y_test', "NearMiss Clean")
    ]

for i, dataset in enumerate(datasets):
    dataset_name = dataset[4]
    file_name = f"Datasets/Val/{dataset_name}.pkl"
    print(f"Loading {dataset_name} Val Sampled Object...")
    with open(file_name, 'rb') as f:
        datasets[i] = pickle.load(f)
print('')
for i, dataset in enumerate(datasets_test):
    dataset_name = dataset[4]
    file_name = f"Datasets/Test/{dataset_name}.pkl"
    print(f"Loading {dataset_name} Test Sampled Object...")
    with open(file_name, 'rb') as f:
        datasets_test[i] = pickle.load(f)