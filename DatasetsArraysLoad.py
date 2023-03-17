import pickle
datasets = [
    ('X_train_adasyn_negations', 'y_train_adasyn_negations', 'x_val_vec_negations', 'y_val', "ADASYN Negations"),
    ('X_train_adasyn_clean', 'y_train_adasyn_clean', 'x_val_vec_clean', 'y_val', "ADASYN Clean"),
    ('x_train_vec_negations', 'y_train_clean', 'x_val_vec_negations', 'y_val', "Original Negations"),
    ('x_train_vec_clean', 'y_train_clean', 'x_val_vec_clean', 'y_val', "Original Clean"),
    ('X_train_ncr_negations', 'y_train_ncr_negations', 'x_val_vec_negations', 'y_val', "NCR Negations"),
    ('X_train_ncr_clean', 'y_train_ncr_clean', 'x_val_vec_clean', 'y_val', "NCR Clean"),
    ('X_train_nm_negations', 'y_train_nm_negations', 'x_val_vec_negations', 'y_val', "NearMiss Negations"),
    ('X_train_nm_clean', 'y_train_nm_clean', 'x_val_vec_clean', 'y_val', "NearMiss Clean")
    ]

for i, dataset in enumerate(datasets):
    dataset_name = dataset[4]
    file_name = f"Datasets/Arrays/Arrays_{dataset_name}.pkl"
    print(f"Loading {dataset_name} Sampled Object...")
    with open(file_name, 'rb') as f:
        datasets[i] = pickle.load(f)
