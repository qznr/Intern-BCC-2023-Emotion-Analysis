import mysql.connector
import optuna
from DatasetsLoad import datasets

if __name__ == '__main__':
    conn = mysql.connector.connect(
        host='localhost',
        user='myuser',
        password='mypassword'
    )
    cursor = conn.cursor()
    cursor.execute("CREATE DATABASE IF NOT EXISTS optuna_svm")
    conn.close()
    for x_train, y_train, x_val, y_val, dataset_name in datasets:
        # Create Optuna study for each datasets with MySQL storage
        study = optuna.create_study(direction="maximize", study_name=f"svm_{dataset_name}", storage=optuna.storages.RDBStorage(url='mysql://myuser:mypassword@localhost/optuna_svm'), load_if_exists=True)