import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('MessageData', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    
    ##data for message distribution by genre 
    genre_counts = df.groupby('genre').count()['message'].sort_values(ascending = False)
    genre_names = list(genre_counts.index.str.capitalize())
    
    ##data for message distribution by category
    category_counts = df[df.columns[4:]].sum(axis = 0).sort_values(ascending = False)
    category_names = list(category_counts.index.str.capitalize().str.replace('_', " "))
    
    
    # create visuals
    
    ##Graph 1: message distribution by category
    graphs = [
        
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message by Categories',
                'yaxis': {
                    'title': "Count"
                }
            }
        },
        
        ##Graph 2: message distribution by genre
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message by Genres',
                'yaxis': {
                    'title': "Count"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()