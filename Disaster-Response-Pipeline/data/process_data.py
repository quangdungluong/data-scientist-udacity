# import libraries
import os
import sys

import pandas as pd
from sqlalchemy import create_engine

sys.path.insert(0, os.getcwd())
from config import CFG


def load_datasets():
    """
    The function `load_datasets` loads and merges two datasets, `messages_df` and `categories_df`, based
    on a specified merge column.
    :return: a merged dataframe that combines the messages dataframe and the categories dataframe.
    """
    messages_df = pd.read_csv(CFG.messages_data_path)
    categories_df = pd.read_csv(CFG.categories_data_path)
    df = pd.merge(messages_df, categories_df, on=CFG.merge_on)
    return df

def process_data(df):
    """
    The function processes a DataFrame by splitting the "categories" column, creating new columns based
    on the split values, converting the values to integers, dropping the original "categories" column,
    and removing duplicate rows.
    
    :param df: The parameter `df` is a pandas DataFrame that contains the data to be processed
    :return: the processed dataframe with the categories split into separate columns and the original
    "categories" column dropped.
    """
    categories = df["categories"].str.split(';', expand=True)
    category_columns = [x.split('-')[0] for x in categories.iloc[0]]
    categories.columns = category_columns
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1]).astype('int')
        
    # replace 2s with 1s in related column
    categories['related'] = categories['related'].replace(to_replace=2, value=1)

    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates()
    return df

def save_data(df):
    """
    The function saves a pandas DataFrame to a SQLite database table.
    
    :param df: The parameter `df` is a pandas DataFrame that contains the data you want to save to a
    SQLite database
    """
    engine = create_engine(f'sqlite:///{CFG.db_path}')
    df.to_sql(f'{CFG.table_name}', engine, index=False, if_exists='replace')

if __name__ == "__main__":
    print("=== Loads data, cleans data, saves data to database ===")
    df = load_datasets()
    df = process_data(df)
    save_data(df)
    print("SUCCESSFULLY")