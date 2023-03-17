import os
from DatasetsLoad import datasets_test
from DatasetsLoad import datasets
from sklearn.svm import SVC
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
            if model_name == "SVM" and dataset_name == dataset_name_in_file:
                # Read in the hyperparameters from the file
                with open(os.path.join(hyperparameter_path, filename), "r") as f:
                    hyperparameters = f.read().splitlines()

                # Convert the hyperparameters to the appropriate data types
                C = float(hyperparameters[0].split(':')[1].strip())
                kernel = hyperparameters[1].split(':')[1].strip()
                
                # Train the SVM model using the hyperparameters
                if kernel == 'linear':
                    model = SVC(C=C, kernel=kernel)
                else:
                    gamma = hyperparameters[2].split(':')[1].strip()
                    model = SVC(C=C, kernel=kernel, gamma=gamma)

                model.fit(X_train, y_train)

                # Evaluate the model on the test set
                y_pred = model.predict(X_test)
                precision = precision_score(y_test, y_pred, average='macro')
                precision_scores[dataset_name].append(precision)

# Print the precision scores for each dataset
with open("ScoresTestSVM.txt", "w") as f:
    for dataset_name, scores in precision_scores.items():
        f.write(f"{dataset_name} - Precision scores: {scores}\n")
