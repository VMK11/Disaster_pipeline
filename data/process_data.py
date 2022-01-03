"""Data Wrangling"""
import numpy    as np
import pandas   as pd
from sqlalchemy import create_engine

"""Text Preprocessing"""
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

"""Miscellaneous"""
import re
import sys
import os

def load_data(messages_filepath, categories_filepath):
    """Load messages.csv & categories.csv and return a merged dataframe.

    Args:
        messages_filepath (String)      : Messages Dataset (.CSV) path.
        categories_filepath (String))   : Categories Dataset (.CSV) path.

    Returns:
        DataFrame: Messages & categories merged dataframe.
    """

    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)

    return pd.merge(messages_df, categories_df, on="id")


def clean_data(df):
    """[summary]

    Args:
        df (DataFrame): DataFrame containing the text messages and the corresponding target labels..

    Returns:
        df (DataFrame): Cleaned dataframe.
    """

    # Load english stop words
    stop = stopwords.words("english")
    
    # Extract target labels from column names.
    categories = df.categories.str.split(pat=';', expand=True)
    firstrow = categories.iloc[0,:]
    category_colnames =  firstrow.apply(lambda x:x[:-2])
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(np.int64)
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], join='inner', axis=1)
    df.drop_duplicates(inplace=True)

    # Remove links & puncuation from the text.
    df['message'] = df.message.str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'url_placeholder')
    df = df[df.message != 'url_placeholder']
    df['message'] = df['message'].str.replace('[^\w\s]','')
    df['message'] = df['message'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    df.drop(columns=['related', 'child_alone'], axis=1, inplace=True)
    return df


def save_data(df, database_filename):
    """

    Args:
        df ([type]): [description]
        database_filename ([type]): [description]
    """

    """
    Saves the data into a database file.
    :param df:
    :param database_filename:
    :return:
    """

    engine = create_engine(f'sqlite:///{database_filename}')
    table_name = os.path.basename(
        database_filename).replace(".db", "") + "_table"
    df.to_sql(table_name, engine, index=False)
    

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
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()