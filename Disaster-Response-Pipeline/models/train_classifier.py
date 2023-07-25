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
    engine = create_engine(f'sqlite:///{CFG.db_path}')
    df = pd.read_sql_table(f"{CFG.table_name}", engine)
    X = df["message"]
    y = df.iloc[:, 4:]
    return X, y

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens=[]
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens

def build_model():
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
    y_pred = model.predict(X_test)
    for index, column in enumerate(y_test):
        print(column, classification_report(y_test[column], y_pred[:, index]))

    
def save_model(model):
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
