import os
from DatasetsLoad import datasets_test
from DatasetsLoad import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Define the path to the hyperparameter files
hyperparameter_path = "Hyperparameter Tuning"

# Initialize an empty list to store the precision scores for each dataset
precision_scores = {}

# Loop over each dataset and hyperparameter file
for dataset in datasets_test:
    X_train, y_train, X_test, y_test, dataset_name = dataset
    precision_scores[dataset_name] = []  # Initialize an empty list for the current dataset
    for filename in os.listdir(hyperparameter_path):
        if filename.endswith(".txt"):
            # Extract the model and dataset names from the filename
            model_name, dataset_name_in_file = filename.split("_")
            dataset_name_in_file = dataset_name_in_file[:-4]  # Remove the ".txt" extension

            # Check if the model and dataset names match the current dataset
            if model_name == "RandomForest" and dataset_name == dataset_name_in_file:
                # Read in the hyperparameters from the file
                with open(os.path.join(hyperparameter_path, filename), "r") as f:
                    hyperparameters = f.read().splitlines()

                # Convert the hyperparameters to the appropriate data types
                n_estimators = int(hyperparameters[0].split(':')[1].strip())
                max_depth = int(hyperparameters[1].split(':')[1].strip())
                min_samples_split = int(hyperparameters[2].split(':')[1].strip())
                min_samples_leaf = int(hyperparameters[3].split(':')[1].strip())

                # Train the random forest model using the hyperparameters
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf
                )
                model.fit(X_train, y_train)

                # Evaluate the model on the test set
                y_pred = model.predict(X_test)
                precision = precision_score(y_test, y_pred, average='macro')
                precision_scores[dataset_name].append(precision)

# Print the precision scores for each dataset
with open("ScoresTestlRf.txt", "w") as f:
    for dataset_name, scores in precision_scores.items():
        f.write(f"{dataset_name} - Precision scores: {scores}\n")