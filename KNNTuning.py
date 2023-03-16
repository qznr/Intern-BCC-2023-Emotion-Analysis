import mysql.connector
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from DatasetsArraysLoad import datasets
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
    conn = mysql.connector.connect(
        host='localhost',
        user='myuser',
        password='mypassword'
    )
    cursor = conn.cursor()
    cursor.execute("CREATE DATABASE IF NOT EXISTS optuna_knn")
    conn.close()
    best_hyperparams = {}
    for x_train, y_train, x_val, y_val, dataset_name in datasets:
        # Create Optuna study for each datasets with MySQL storage
        study = optuna.load_study(study_name=f"knn_{dataset_name}", storage=optuna.storages.RDBStorage(url='mysql://myuser:mypassword@localhost/optuna_knn'))
        study.optimize(lambda trial: objective(trial, x_train, y_train, x_val, y_val), callbacks=[MaxTrialsCallback(100, states=(TrialState.COMPLETE,))], show_progress_bar=True)
        best_hyperparams[dataset_name] = study.best_params
        print(f"Best hyperparameters for {dataset_name}: {best_hyperparams[dataset_name]}")
        with open(f'Hyperparameter Tuning/KNN_{dataset_name}.txt', 'w') as f:
            f.write(f"n_neighbors: {best_hyperparams[dataset_name]['n_neighbors']}\n")
            f.write(f"weights: {best_hyperparams[dataset_name]['weights']}\n")
            f.write(f"algorithm: {best_hyperparams[dataset_name]['algorithm']}\n")
