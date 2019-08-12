import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads csv files into dataframes
    :param messages_filepath: Path to messages csv
    :param categories_filepath: Path to categories csv
    :return: both the dataframes
    """

    df_cat = pd.read_csv(categories_filepath)
    df_mess = pd.read_csv(messages_filepath, usecols=[0, 1, 3])

    return df_cat, df_mess


def clean_data(df_cat, df_mess):
    """
    Cleans the data to feed to the model
    :param df_cat: categories dataframe
    :param df_mess: messages dataframe
    :return: merged and cleaned dataframe
    """

    # Create a list of the columns for our labels for multi-label classification
    label_cols = [i[:-2] for i in df_cat['categories'][0].split(';')]

    # Split and expand the categories column into multiple columns. 
    # See: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.split.html
    df_cat['categories'] = df_cat['categories'].apply(lambda x: ', '.join([i[-1] for i in x.split(';')]))
    df_cat[label_cols] = df_cat['categories'].str.split(',', expand=True)

    # Drop the no longer needed categories column. Also, 'child_alone' has all values set to 0
    df_cat.drop(['child_alone', 'categories'], axis=1, inplace=True)

    # Type cast all the values to int32
    df_cat = df_cat.astype('int32')

    # Merge the datasets on 'id' column
    df = pd.merge(df_mess, df_cat, on='id')

    # 'related' column should have value as 0 or 1, but it has 0, 1 and 2. So, we drop rows with value == 2
    df = df.loc[df['related'].isin([0, 1]), :]

    # And finally, we drop rows with duplicate 'id'
    df.drop_duplicates('id', inplace=True)

    return df


def save_data(df, database_filename):
    """
    Saves dataframe to database
    :param df: cleaned dataframe
    :param database_filename: path to store database
    """

    engine = create_engine('sqlite:///{}'.format(database_filename))

    # Extract file name from filepath
    db_file_name = database_filename.split('\\')[-1]

    table_name = db_file_name.split('.')[0]
    df.to_sql(table_name, engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, categories_filepath))

        df_cat, df_mess = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df_cat, df_mess)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories datasets as the first and second argument '
              'respectively, as well as the filepath of the database to save the cleaned data to as the third argument.'
              '\n\nExample: python process_data.py disaster_messages.csv disaster_categories.csv database.db')


if __name__ == '__main__':
    main()
