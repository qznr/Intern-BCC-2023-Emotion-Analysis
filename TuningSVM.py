import optuna
import sqlite3
import os
from tqdm import tqdm
from DatasetsLoad import datasets
from sklearn.svm import SVC
from sklearn.metrics import precision_score

def objective(trial, x_train, y_train, x_val, y_val):
    svm_c = trial.suggest_float("C", 1e-4, 10, log=True)
    svm_kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])
    svm_gamma = trial.suggest_categorical("gamma", ["scale", "auto"]) if svm_kernel in ["rbf", "poly", "sigmoid"] else "scale"
    svm_model = SVC(C=svm_c, kernel=svm_kernel, gamma=svm_gamma)
    svm_model.fit(x_train, y_train)
    y_pred = svm_model.predict(x_val)
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
        with open(f'Hyperparameter Tuning/SVM_{dataset_name}.txt', 'w') as f:
            f.write(f"C: {best_hyperparams[dataset_name]['C']}\n")
            f.write(f"Kernel: {best_hyperparams[dataset_name]['kernel']}\n")
            f.write(f"Gamma: {best_hyperparams[dataset_name]['gamma']}\n")