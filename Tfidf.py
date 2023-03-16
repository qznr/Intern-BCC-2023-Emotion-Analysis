import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer 

# Import data
from DataPreprocessing import df_train_clean_negations, df_train_clean, val, test

# Set paths
MODEL_PATH = 'tf-idf/models/'
DATA_PATH = 'tf-idf/data/'

# Create directories
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

# Define function to load or save objects using pickle
def load_or_save(obj, filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
    else:
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)
    return obj

# Define TF-IDF objects
vec_clean = TfidfVectorizer()
vec_negations = TfidfVectorizer()

# Define file names for TF-IDF models
vec_clean_filename = MODEL_PATH + 'vec_clean.pkl'
vec_negations_filename = MODEL_PATH + 'vec_negations.pkl'

# Load or save TF-IDF models
vec_clean = load_or_save(vec_clean, vec_clean_filename)
vec_negations = load_or_save(vec_negations, vec_negations_filename)

# Define data
x_train_negations = df_train_clean_negations['Deskripsi'] 
x_train_clean = df_train_clean['Deskripsi']
x_val = val['Deskripsi']
x_test = test['Deskripsi']
y_train_clean = df_train_clean['Emosi'] 
y_val = val['Emosi']
y_test = test['Emosi']

# Define file names for transformed data
x_train_vec_clean_filename = DATA_PATH + 'x_train_vec_clean.pkl'
x_train_vec_negations_filename = DATA_PATH + 'x_train_vec_negations.pkl'
x_val_vec_clean_filename = DATA_PATH + 'x_val_vec_clean.pkl'
x_val_vec_negations_filename = DATA_PATH + 'x_val_vec_negations.pkl'
x_test_vec_clean_filename = DATA_PATH + 'x_test_vec_clean.pkl'
x_test_vec_negations_filename = DATA_PATH + 'x_test_vec_negations.pkl'

# Load or save transformed data
x_train_vec_clean = load_or_save(vec_clean.fit_transform(x_train_clean), x_train_vec_clean_filename)
x_train_vec_negations = load_or_save(vec_negations.fit_transform(x_train_negations), x_train_vec_negations_filename)
x_val_vec_clean = load_or_save(vec_clean.transform(x_val), x_val_vec_clean_filename)
x_val_vec_negations = load_or_save(vec_negations.transform(x_val), x_val_vec_negations_filename)
x_test_vec_clean = load_or_save(vec_clean.transform(x_test), x_test_vec_clean_filename)
x_test_vec_negations = load_or_save(vec_negations.transform(x_test), x_test_vec_negations_filename)