import optuna
import sqlite3
import os
from tqdm import tqdm
from DatasetsDenseLoad import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score

def objective(trial, x_train, y_train, x_val, y_val):
    n_neighbors = trial.suggest_int("n_neighbors", 1, 20)
    p = trial.suggest_int("p", 1, 3)
    leaf_size = trial.suggest_int("leaf_size", 10, 100)
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors, p=p, leaf_size=leaf_size)
    knn_model.fit(x_train, y_train)
    y_pred = knn_model.predict(x_val)
    precision = precision_score(y_val, y_pred, average='macro')
    return precision

if __name__ == '__main__':
    if not os.path.exists('Hyperparameter Tuning'):
        os.makedirs('Hyperparameter Tuning')
    conn = sqlite3.connect('optuna.db')
    best_hyperparams = {}
    for x_train, y_train, x_val, y_val, dataset_name in tqdm(datasets):
        # Create Optuna study for each datasets with SQLite storage
        study = optuna.create_study(direction="maximize", storage=optuna.storages.RDBStorage(url='sqlite:///optuna.db'), load_if_exists=True)
        with tqdm(total=100, desc=f"Optimizing {dataset_name}") as progress_bar:
            def objective_with_progress_bar(trial):
                progress_bar.update(1)
                return objective(trial, x_train, y_train, x_val, y_val)
            study.optimize(objective_with_progress_bar, n_trials=100)
        best_hyperparams[dataset_name] = study.best_params
        print(f"Best hyperparameters for {dataset_name}: {best_hyperparams[dataset_name]}")
        with open(f'Hyperparameter Tuning/KNN_{dataset_name}.txt', 'w') as f:
            f.write(f"n_neighbors: {best_hyperparams[dataset_name]['n_neighbors']}\n")
            f.write(f"p: {best_hyperparams[dataset_name]['p']}\n")
