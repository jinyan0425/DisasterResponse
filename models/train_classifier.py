import sys

import pandas as pd
import numpy as np

import pickle

from sqlalchemy import create_engine

import warnings
warnings.filterwarnings('ignore')


# IMPORT ntlk LIBRARIES
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# IMPORT sklearn LIBRARIES
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV



#LOAD DATA
def load_data(database_filepath):
    
    '''
    Parameters
    ----------
    df : DATAFRAME,the dataframe.

    Returns
    -------
    DATAFRAME, the first five rows of the loaded dataframe.
    '''
    
    # load data from database
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('MessageData', engine)
    
    #drop the entries wit label 2 in 'related_category' (N = 188 out of 26,180), whic is likely to subject to recording error
    df = df[df.related != 2]
    
    #drop the 'child_alone' category, which has not relevant cases in the sample (i.e., all zeros)
    df.drop('child_alone', axis = 1, inplace = True)
    
    #create X (the feature array)
    X = df['message']
    
    #create y (the target variables array)
    y = df[df.columns[4:]]
   
    #create the names of cateogries
    category_names = y.columns
    
    return X, y, category_names


#CREATE A TOKENIZE FUNCTION TO PROCESS TEXT DATA
def tokenize(text):
    
    '''
    Parameters
    ----------
    text : STRING, messages/texts needed to tokenized.

    Returns
    -------
    cleaned_tokens : LIST, a list of processed words (lemmatized, lowercased) extracted from the original text
    '''
    
    #tokenize words in the texts
    token = word_tokenize(text)
    
    #lemmatize words in the texts
    lemmatizer = WordNetLemmatizer()
    
    #create a list to store processed words
    cleaned_tokens = []
    
    for tok in token:
        #lower the cases of the words and remove any leading and trailing characters
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        cleaned_tokens.append(clean_tok)
    
    return cleaned_tokens



#BUILD A ML PIPELINE (use LogisticRegression as the classifier)
def build_model():
    
    '''
    Parameters
    ----------
    None.

    Returns
    -------
    DICTIONARY, the model estimator and paramaters for 
    '''
    

    pipeline = Pipeline([

    ('vect', CountVectorizer(tokenizer = tokenize)),
    
    ('tfidf', TfidfTransformer()),
    
    #set the class_weight to be 'balanced' as most categories have unbalanced classes,
    #set n_jobs to be -1 to use all processors
    ('cfl', MultiOutputClassifier(LogisticRegression
                                  (class_weight = 'balanced', n_jobs = -1)
                                  ))
    ])
    
    
    #set paramaters
    parameters = {
        ##tune the regularization strength, default is 1
        'cfl__estimator__C': np.arange(0,1.2,0.2), 
    
        ##set the class_weight to be 'balanced' as most categories have unbalanced classes,
        ##default is None, which treated all classes equally
        'cfl__estimator__class_weight':['balanced'],
    
        ##increase the max iterations, default is 100
        'cfl__estimator__max_iter': [100, 500, 1000],
    
        ##set n_jos to be -1 to use all processors
        'cfl__estimator__n_jobs':[-1]
        }
    
    #set scoring to be f1_micro because the classes have unbalanced weights
    #set n_jobs to be -1 to use all processors
    cv_model = GridSearchCV(pipeline, 
                               param_grid=parameters, 
                               scoring='f1_micro', n_jobs=-1)


    return cv_model 
    


#EVALUATING THE MODEL USING F1 score
def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    Parameters
    ----------
    model : CLASSIFIER, model for classification.
    X_test : 1D ARRAY, features in the testing dataset
    Y_test : 2D ARRAY, target variables in the testing dataset
    category_names : LIST, category names of the messages being classified

    Returns
    -------
    STRING, Classficiation table.
    '''

    y_pred = model.predict(X_test)
    y_true = Y_test.values
    
    print(classification_report(y_true, y_pred, target_names=category_names))



#EVALUATING THE MODEL WITH DIFFERENT CLASSIFICATION THRESHOLDS
##create a function to evaluate different thresholds for classfication (default = 0.5) as some message categories have very skewed data
def evaluate_thresholds(model, X_test, Y_test, thresholds):
    
    '''
    Parameters
    ----------
    model : CLASSIFIER, model for classification.
    X_test : 1D ARRAY, features in the testing dataset
    Y_test : 2D ARRAY, target variables in the testing dataset
    thresholds : LIST, thresholds will be used for evaluation

    Returns
    -------
    LIST, evaluation results on different thresholds

    '''
    #predct probabilities
    y_pred_proba = model.predict_proba(X_test)
    y_true = Y_test.values
    
    #create a list to store evaluations
    threshold_results = []
    
    #loop through all thresholds
    for th in thresholds:
        y_pred_new_threshold = np.zeros(y_true.shape)
        
        #loop through each message category
        for i, predicts in enumerate(y_pred_proba):
            y_pred_new_threshold_i = (predicts[:,1] >= th).astype(int)
            y_pred_new_threshold[:,i] = y_pred_new_threshold_i 
        
        #calcualte f1 scores(micro and weighted) for each threshold
        f1_micro = f1_score(y_true, y_pred_new_threshold, average = 'micro')    
        f1_weighted = f1_score(y_true, y_pred_new_threshold, average = 'weighted')

        #create tuples for each threshold and store it in the result list
        threshold_results.append(("threshold: {}".format(th.round(2)), 
                                  "f1_micro: {}".format(f1_micro.round(3)),
                                  "f1_weighted: {}".format(f1_weighted.round(3))
                                 ))
        
    return threshold_results



#SAVE MODEL
def save_model(model, model_filepath):
    
    '''
    Parameters
    ----------
    model : DICTIONARY, the classfication model estimator and paramaters that yield the best results.
    model_filepath : STRING, the pickle file path.

    Returns
    -------
    None.

    '''
    
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        thresholds = np.arange(0.2,0.8,0.1)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        
        print('Evaluating thresholds of the classification...')
        evaluate_thresholds(model, X_test, Y_test, thresholds)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the file path of the disaster messages database '\
              'as the first argument and the file path of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
