import os
import pickle
import sys

import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

sys.path.insert(0, os.getcwd())
from config import CFG

nltk.download(['punkt', 'wordnet'])


def load_data():
    """
    The function `load_data` loads data from a SQLite database table and returns the message column as X
    and the remaining columns as y.
    :return: two variables, X and y. X is a pandas Series containing the "message" column from the
    loaded data, and y is a pandas DataFrame containing all columns from the 4th column onwards.
    """
    engine = create_engine(f'sqlite:///{CFG.db_path}')
    df = pd.read_sql_table(f"{CFG.table_name}", engine)
    X = df["message"]
    y = df.iloc[:, 4:]
    return X, y

def tokenize(text):
    """
    The `tokenize` function takes in a text and returns a list of lemmatized and lowercased tokens.
    
    :param text: The `text` parameter is a string that represents the input text that needs to be
    tokenized
    :return: a list of clean tokens.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens=[]
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens

def build_model():
    """
    The function `build_model` returns a pipeline with a grid search cross-validation object for
    hyperparameter tuning.
    :return: a GridSearchCV object.
    """
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    parameters = {
    'clf__estimator__n_estimators' : [100, 150],
    'clf__estimator__learning_rate': [0.5, 1.00]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, y_test):
    """
    The function evaluates a machine learning model by predicting the target variable for a given test
    dataset and printing the classification report for each column of the target variable.
    
    :param model: The `model` parameter is the trained machine learning model that you want to evaluate.
    It could be any model that has a `predict` method, such as a classifier or a regressor
    :param X_test: X_test is the input data that will be used to evaluate the model's performance. It
    should be a matrix or dataframe containing the features or independent variables. Each row
    represents a sample or instance, and each column represents a feature
    :param y_test: The y_test parameter is the true labels or target values for the test dataset. It is
    a dataframe or array-like object containing the actual values of the target variable for each sample
    in the test dataset
    """
    y_pred = model.predict(X_test)
    for index, column in enumerate(y_test):
        print(column, classification_report(y_test[column], y_pred[:, index]))

    
def save_model(model):
    """
    The function saves a machine learning model to a file using pickle.
    
    :param model: The `model` parameter is the machine learning model that you want to save
    """
    pickle.dump(model, open(CFG.model_path, "wb"))

if __name__ == "__main__":
    print("=== Train model ===")
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    model = build_model()
    model.fit(X_train, y_train)

    evaluate_model(model, X_test, y_test)
    save_model(model)
    print("SUCCESSFULLY")
