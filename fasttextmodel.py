import fasttext
import re
from xmlrpc.client import Boolean
import numpy as np
import pandas as pd
import sklearn.metrics
import xgboost as xgb


#data from https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset

def read_data(data_path: str, ftmodel_path: str) -> pd.DataFrame:
    """Read hate speech data, process so it can be used with fasttext
    
    Args:
        path (str): location on disk
        
        
    Returns:
        pd.DataFrame: a dataframe of tweets labeled as hate speech or not, 
        and the fasttext vector features 
    """
    df = pd.read_csv(data_path)
    df['hate'] = df['class'].apply(is_hate)
    df['tweets_processed'] = df['tweet'].apply(preprocessor)
    #apply fasttext to create a new feature column
    ft_model = fasttext.load_model(ftmodel_path)
    df['fasttext']=df['tweets_processed'].apply(lambda x: ft_model.get_sentence_vector(x))
    #create 300 cols of data from 300d fasttext vector
    data = []
    for row in df['fasttext']:
        data.append(row)
    df_transform = pd.DataFrame(data)
    df_transform['hate'] = df['hate']
    return df_transform

def is_hate(x: int) -> Boolean:
    if x == 2:
        return 0
    else:
        return 1

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:)|(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text


def train_test_split(df: pd.DataFrame, seed: int = 34892) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Split data into training set and test set

    Args:
        df (pd.DataFrame): dataframe containing the data
        seed (int): seeding for random_state in train_test_split()
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: training features, training labels,
        test features, test labels
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

def assess_binary(actual: pd.DataFrame, prediction: pd.DataFrame) -> float:
    """Assess quality of model

    Args:
        actual (pd.DataFrame): dataframe of true values in test set
        prediction (pd.DataFrame): dataframe of predicted values for test set

    Returns:
        float: F1 score of model
    
    """

    #F1 score
    F_1 = sklearn.metrics.f1_score(actual,prediction)
    print(f"The F1 score is {F_1}")
    return F_1


if __name__=='__main__':
    print('read in data')
    df=read_data(data_path='data/labeled_data.csv', ftmodel_path='data/cc.en.300.bin')
    print('data has been read')
    labels = ['hate']
    train_set, test_set = train_test_split(df)
    train_features = train_set.drop(columns=labels)
    test_features = test_set.drop(columns=labels)
    print('data has been split')
    model = train(train_features, train_set[labels], model_choice=xgb.XGBClassifier())
    print('model has been created')
    test_features['predictions']=predict(model, test_features)
    print('predictions have been made')
    assess_binary(test_set[labels], test_features['predictions'])

    