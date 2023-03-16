import mysql.connector
import os
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from DatasetsLoad import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

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
    conn = mysql.connector.connect(
        host='localhost',
        user='myuser',
        password='mypassword'
    )
    cursor = conn.cursor()
    cursor.execute("CREATE DATABASE IF NOT EXISTS optuna_rf")
    conn.close()
    if not os.path.exists('Hyperparameter Tuning'):
        os.makedirs('Hyperparameter Tuning')
    best_hyperparams = {}
    for x_train, y_train, x_val, y_val, dataset_name in datasets:
        # Create Optuna study for each datasets with MySQL storage
        study = optuna.load_study(study_name = f"rf_{dataset_name}", storage=optuna.storages.RDBStorage(url='mysql://myuser:mypassword@localhost/optuna_rf'))
        study.optimize(lambda trial: objective(trial, x_train, y_train, x_val, y_val), callbacks=[MaxTrialsCallback(500, states=(TrialState.COMPLETE,))], show_progress_bar=True)
        best_hyperparams[dataset_name] = study.best_params
        print(f"Best hyperparameters for {dataset_name}: {best_hyperparams[dataset_name]}")
        with open(f'Hyperparameter Tuning/RandomForest_{dataset_name}.txt', 'w') as f:
            f.write(f"n_estimators: {best_hyperparams[dataset_name]['n_estimators']}\n")
            f.write(f"max_depth: {best_hyperparams[dataset_name]['max_depth']}\n")
            f.write(f"min_samples_split: {best_hyperparams[dataset_name]['min_samples_split']}\n")
            f.write(f"min_samples_leaf: {best_hyperparams[dataset_name]['min_samples_leaf']}\n")

