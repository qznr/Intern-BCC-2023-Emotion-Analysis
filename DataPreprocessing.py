import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def define_stopwords(negations=True):
    new_stopwords = ['feel','feeling','feelings','feels','felt','im','id','ive','dont','couldnt','cant','neednt','havent','isnt','didnt','wont','hasnt','wasnt','wont','shant','hadnt','arent','werent','wouldnt','would']
    stopwords_ = stopwords.words('english')
    stopwords_.extend(new_stopwords)
    stopwords_list = stopwords_
    if negations:
        common_negations = ['no','not','dont','don','isnt','isn','arent','aren','didnt','didn','cant','wouldnt','wouldn','weren','werent','wasn','wasnt','wont']
        stopwords_list = [word for word in stopwords_list if word not in common_negations]
    return stopwords_list

def words_stemmer(to_be_stemmed):
  stemmer = PorterStemmer()
  stemmed_words = []
  for word in to_be_stemmed:
    stemmed_word = stemmer.stem(word)
    stemmed_words.append(stemmed_word)
  return ' '.join(stemmed_words)

val = pd.read_csv('Datasets/val.txt', delimiter = ";", header = None)
val.columns = ['Deskripsi', 'Emosi']
test = pd.read_csv('Datasets/test.txt', delimiter = ";", header = None)
test.columns = ['Deskripsi', 'Emosi']

if not os.path.exists('Datasets'):
        os.makedirs('Datasets')

if os.path.exists('Datasets/df_train_clean_stemmed.csv') and os.path.exists('Datasets/df_train_clean_negations_stemmed.csv'):
    df_train_clean = pd.read_csv('Datasets/df_train_clean_stemmed.csv')
    df_train_clean_negations = pd.read_csv('Datasets/df_train_clean_negations_stemmed.csv')
else:
    # Read CSVs
    test = pd.read_csv('Datasets/test.txt', delimiter = ";", header = None)
    test.columns = ['Deskripsi', 'Emosi']
    train = pd.read_csv('Datasets/train.txt', delimiter = ";", header = None)
    train.columns = ['Deskripsi', 'Emosi']
    val = pd.read_csv('Datasets/val.txt', delimiter = ";", header = None)
    val.columns = ['Deskripsi', 'Emosi']
    # Copy Dataframe
    df_train_clean = train.copy()
    # Clear Duplicates
    df_train_clean = df_train_clean.drop_duplicates(subset = ['Deskripsi'])
    df_train_clean.describe()
    # Stopwords
    nltk.download('stopwords')
    # Make new df with negations
    df_train_clean_negations = df_train_clean.copy()
    df_train_clean_negations['Deskripsi'] = df_train_clean['Deskripsi'].apply(lambda x: ' '.join([word for word in x.split() if word not in define_stopwords(negations=True)]))
    df_train_clean['Deskripsi'] = df_train_clean['Deskripsi'].apply(lambda x: ' '.join([word for word in x.split() if word not in define_stopwords(negations=False)]))
    # Stemming
    df_train_clean_negations['Deskripsi'] = df_train_clean_negations['Deskripsi'].apply(lambda x : words_stemmer(x.split()))
    df_train_clean['Deskripsi'] = df_train_clean['Deskripsi'].apply(lambda x : words_stemmer(x.split()))
    df_train_clean.to_csv('Datasets/df_train_clean_stemmed.csv', index=False)
    df_train_clean_negations.to_csv('Datasets/df_train_clean_negations_stemmed.csv', index=False)