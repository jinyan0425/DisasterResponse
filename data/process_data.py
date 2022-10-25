import sys
import pandas as pd
from sqlalchemy import create_engine



#LOAD DATA
def load_data(messages_filepath, categories_filepath):
    
    '''
    Parameters
    ----------
    messages_filepath : STRING, url path.
    categories_filepath : STRING, url path.

    Returns
    -------
    DATAFRAME, the merged and ready-t-clean dataframe.
    '''
   
    #load messages dataset
    messages = pd.read_csv(messages_filepath) 
    
    #load cateogries dataset
    categories = pd.read_csv(categories_filepath) 
    
    
    #merge two datasets
    df = messages.merge(categories, on = 'id', how = 'inner') 
    
    
    #split categories into seperate category columns
    
    ##create a temp dataframe of the 36 individual category columns ('categories_temp')
    categories_temp = df.categories.str.split(';', expand = True)
    
    ##extract a list of new loumn names fore cateories
    category_colnames = list(categories_temp.iloc[1].apply(lambda x:x.split('-')[0]))
    
    ## rename the columns of 'categories'
    categories_temp.columns = category_colnames
    
    #convert category values to numbers 0 or 1
    for column in categories_temp:
        categories_temp [column] = categories_temp [column].apply(lambda x:x.split('-')[1])
    
    categories_temp = categories_temp.astype('int64')
    
    
    #replace categories column in df with new category columns
    
    ##drop the original 'categories' column
    df.drop('categories',axis = 1, inplace = True)
   
    ##concatenate the original dataframe with the new 'categories_temp' dataframe
    df = pd.concat([df, categories_temp],axis = 1)


    return df



#CLEAN DATA
def clean_data(df):
    
    '''
    Parameters
    ----------
    df : DATAFRAME,the dataframe.

    Returns
    -------
    DATAFRAME, the cleaned dataframe.
    '''
    
    #drop duplicates (based on id)
    df.drop_duplicates(subset = ['id'], inplace = True)
    
    return df



#SAVE DATA
def save_data(df, database_filepath):
    
    '''
    Parameters
    ----------
    df : DATAFRAME, the cleaned dataframe.
        
    database_filename : STRING, the database path.

    Returns
    -------
    None.
    '''
    
    #save the clean dataset into an sqlite database.
    engine = create_engine('sqlite:///'+ database_filepath)
    df.to_sql('MessageData', engine, index=False)
    


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
        print('Please provide the file paths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the file path of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
