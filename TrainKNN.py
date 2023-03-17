import os
from DatasetsLoad import datasets_test
from DatasetsLoad import datasets
from sklearn.neighbors import KNeighborsClassifier
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
            if model_name == "KNN" and dataset_name == dataset_name_in_file:
                # Read in the hyperparameters from the file
                with open(os.path.join(hyperparameter_path, filename), "r") as f:
                    hyperparameters = f.read().splitlines()

                # Convert the hyperparameters to the appropriate data types
                n_neighbors = int(hyperparameters[0].split(':')[1].strip())
                p = int(hyperparameters[1].split(':')[1].strip())

                # Train the KNN model using the hyperparameters
                model = KNeighborsClassifier(n_neighbors=n_neighbors, p=p)
                model.fit(X_train.toarray(), y_train)

                # Evaluate the model on the test set
                y_pred = model.predict(X_test.toarray())
                precision = precision_score(y_test, y_pred, average='macro')
                precision_scores[dataset_name].append(precision)

# Print the precision scores for each dataset
with open("ScoresTestKNN.txt", "w") as f:
    for dataset_name, scores in precision_scores.items():
        f.write(f"{dataset_name} - Precision scores: {scores}\n")