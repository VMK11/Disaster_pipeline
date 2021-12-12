import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load messages.csv & categories.csv and return a merged dataframe
    :param messages_filepath:
    :param categories_filepath:
    :return: merged_df
    """
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)

    return pd.merge(messages_df, categories_df, on="id")


def clean_data(df):
    """
    1. Splits the strings into different categories.
    2. Extract column names from the first row.
    3. Drop categories columns from the initial dataframe
    4. Concatenate the dataframe with the categories dataframe
    5. Drop the duplicate records.

    :param df:
    :return df:
    """
    categories = df.categories.str.split(pat=';', expand=True)
    firstrow = categories.iloc[0,:]
    category_colnames =  list(map(lambda i: i[ : -2], firstrow)) 
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(np.int)
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], join='inner', axis=1)
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Saves the data into a database file.
    :param df:
    :param database_filename:
    :return:
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('df', engine, index=False)


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