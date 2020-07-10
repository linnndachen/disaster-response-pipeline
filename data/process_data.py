import sys
import pandas as pd 
import numpy as np
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, left_on='id', right_on='id', how='outer')
    return df

def clean_data(df):
    #split categories into separate category columns
    categories = df['categories'].str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    #convert category values to nums 0 & 1
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = pd.to_numeric(categories[column])

    #reaplace the category column with the new columns
    df.drop(['categories'], inplace=True, axis=1)
    df = pd.concat([df, categories], axis=1)

    #remove duplicates
    if df.duplicated().sum() > 0:
        df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename, pool_pre_ping=True)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()