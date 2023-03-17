import mysql.connector
import os
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from DatasetsLoad import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score

def objective(trial, x_train, y_train, x_val, y_val):
    nb_var_smoothing = trial.suggest_float("var_smoothing", 1e-12, 1e-5)
    nb_model = GaussianNB(var_smoothing=nb_var_smoothing)
    nb_model.fit(x_train, y_train)
    y_pred = nb_model.predict(x_val)
    precision = precision_score(y_val, y_pred, average='macro')
    return precision

if __name__ == '__main__':
    conn = mysql.connector.connect(
        host='localhost',
        user='myuser',
        password='mypassword'
    )
    cursor = conn.cursor()
    cursor.execute("CREATE DATABASE IF NOT EXISTS optuna_nb")
    conn.close()
    if not os.path.exists('Hyperparameter Tuning'):
        os.makedirs('Hyperparameter Tuning')
    best_hyperparams = {}
    for x_train, y_train, x_val, y_val, dataset_name in datasets:
        # Create Optuna study for each datasets with MySQL storage
        study = optuna.load_study(study_name = f"nb_{dataset_name}", storage=optuna.storages.RDBStorage(url='mysql://myuser:mypassword@localhost/optuna_nb'))
        study.optimize(lambda trial: objective(trial, x_train, y_train, x_val, y_val), n_trials = 50, show_progress_bar=True)
        best_hyperparams[dataset_name] = study.best_params
        print(f"Best hyperparameters for {dataset_name}: {best_hyperparams[dataset_name]}")
        with open(f'Hyperparameter Tuning/NaiveBayes_{dataset_name}.txt', 'w') as f:
            f.write(f"var_smoothing: {best_hyperparams[dataset_name]['var_smoothing']}\n")
