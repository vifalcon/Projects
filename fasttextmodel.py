import fasttext
import re
from xmlrpc.client import Boolean
import numpy as np
import pandas as pd
import sklearn.metrics
import xgboost as xgb
import sklearn.metrics 
import sklearn.naive_bayes
import tensorflow as tf

'''
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data
'''

def read_data(path: str) -> pd.DataFrame:
    """Read hate speech data, process so it can be used with fasttext
    
    Args:
        path (str): location on disk
        
        
    Returns:
        pd.DataFrame: a dataframe of tweets labeled as hate speech or not
    """
    df = pd.read_csv(path)
    df['hate'] = df['class'].apply(is_offensive)
    df['tweets_processed'] = df['tweet'].apply(process_tweet)
    
    #apply fasttext to create a new feature column
    
    df['fasttext']=df['tweets_processed'].apply()
    
    
    
    #df = df.dropna(subset=['nst'])
    return df

def is_offensive(x: int) -> Boolean:
    if x == 2:
        return 0
    else:
        return 1

def process_tweet(s: str) -> str:
    #re.sub()
    pass

def fasttext_vec(s: str):
    model = fasttext.load_model('cc.en.300.bin')
    return fasttext.get_sentence_vector(s)

def preprocess(words):
    words = tf.strings.regex_replace(words, b"<br\s*/?>", b" ")
    words = tf.strings.regex_replace(words, b"[^a-zA-Z]", b" ")
    words = tf.strings.lower(words)
    words = tf.strings.split(words)
    return words


def train_test_split(df: pd.DataFrame, seed: int = 34892) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training set and test test

    Args:
        df (pd.DataFrame): dataframe containing the data
        seed (int): seeding for random_state in train_test_split()
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: training set, test set
    """
    train_set, test_set = sklearn.model_selection.train_test_split(df, test_size=0.3, random_state=seed)
    return(train_set, test_set)

def train(features: pd.DataFrame, labels: pd.DataFrame, model_choice):
    """Train an xgboost model

    Args:
        features (pd.DataFrame): a dataframe of features
        labels (pd.Series): a pandas column of labels
        model_choice: a specific classifier such as xgb.XGBRegressor()
    
    Returns:
        model: a model of the given classifier
    
    """
    model = model_choice
    model.fit(features, labels)
    
    return model

def predict(model, test_set: pd.DataFrame) -> np.ndarray:
    """Predictions on test data

    Args:
        model: classifier or regressor being used
        features (pd.DataFrame): a dataframe of features matching the training columns 
        (from the test set)

    Returns:
        np.ndarray: prediction for each earthquake
    
    """

    return model.predict(test_set)



#fasttext.util.download_model('en',if_exists='ignore')

if __name__=='__main__':
    model = fasttext.load_model('cc.en.300.bin')

    #dir(model)
    #print(model.get_nearest_neighbors('good'))
    #model.get_word_vector('example')
    #model.get_analogies()
    
    