# import libraries
import os
import sys

import pandas as pd
from sqlalchemy import create_engine

sys.path.insert(0, os.getcwd())
from config import CFG


def load_datasets():
    messages_df = pd.read_csv(CFG.messages_data_path)
    categories_df = pd.read_csv(CFG.categories_data_path)
    df = pd.merge(messages_df, categories_df, on=CFG.merge_on)
    return df

def process_data(df):
    categories = df["categories"].str.split(';', expand=True)
    category_columns = [x.split('-')[0] for x in categories.iloc[0]]
    categories.columns = category_columns
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1]).astype('int')

    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates()
    return df

def save_data(df):
    engine = create_engine(f'sqlite:///{CFG.db_path}')
    df.to_sql(f'{CFG.table_name}', engine, index=False)

if __name__ == "__main__":
    print("=== Loads data, cleans data, saves data to database ===")
    df = load_datasets()
    df = process_data(df)
    save_data(df)
    print("SUCCESSFULLY")