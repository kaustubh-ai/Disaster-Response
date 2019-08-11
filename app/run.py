import re
import json
import numpy as np
import pandas as pd
import spacy
import plotly
from plotly.graph_objs import Bar
from pprint import pprint
from flask import Flask
from flask import render_template, request
import pickle
from collections import Counter
from sqlalchemy import create_engine


nlp = spacy.load('en_core_web_sm')
app = Flask(__name__)


def tokenise(row):
    """
    Cleans the dataframe row by normalising and lemmatisation
    :param row: row of dataframe
    :return: cleaned row of dataframe
    """

    # Creating doc object for SpaCy
    doc = nlp(re.sub(r"[^a-zA-Z0-9]", ' ', row.lower()))
    clean_row = []

    for tok in doc:
        clean_tok = tok.lemma_.strip()  # Lemmatisation
        if len(clean_tok) > 1 and clean_tok != '-PRON-' and not tok.is_stop:
            clean_row.append(clean_tok)

    return clean_row


# load data
engine = create_engine('sqlite:///../data/database.db')
df = pd.read_sql_table('database', engine)

# load model
model = pickle.load(open('../models/model_svc.pkl', 'rb'))


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
# @app.route('/index')
def index():
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']  # message count based on genre
    genre_names = list(genre_counts.index)  # genre names

    cat_p = df[df.columns[4:]].sum() / len(df)  # proportion based on categories
    cat_p = cat_p.sort_values(ascending=False)  # largest bar will be on left
    cats = list(cat_p.index)  # category names

    words_with_repetition = []  # will contain all words words with repetition
    for text in df['message'].values:
        tokenized_ = tokenise(text)
        words_with_repetition.extend(tokenized_)

    word_count_dict = Counter(words_with_repetition)  # dictionary containing word count for all words

    sorted_word_count_dict = dict(sorted(word_count_dict.items(),
                                         key=lambda kv: kv[1],
                                         reverse=True))  # sort dictionary by\
    # values
    top, top_10 = 0, {}

    for k, v in sorted_word_count_dict.items():
        top_10[k] = v
        top += 1
        if top == 10:
            break
    words = list(top_10.keys())
    pprint(words)
    count_props = 100 * np.array(list(top_10.values())) / df.shape[0]

    # create visuals
    figures = [{'data': [Bar(x=genre_names, y=genre_counts)],
                'layout': {'title': 'Distribution of Message Genres',
                           'yaxis': {'title': "Count"},
                           'xaxis': {'title': "Genre"}}},

               {'data': [Bar(x=cats, y=cat_p)],
                'layout': {'title': 'Proportion of Messages <br> by Category',
                           'yaxis': {'title': "Proportion", 'automargin': True},
                           'xaxis': {'title': "Category",
                                     'tickangle': -40,
                                     'automargin': True}}},

               {'data': [Bar(x=words, y=count_props)],
                'layout': {'title': 'Frequency of top 10 words <br> as percentage',
                           'yaxis': {'title': 'Occurrence<br>(Out of 100)',
                                     'automargin': True},
                           'xaxis': {'title': 'Top 10 words',
                                     'automargin': True}}}]

    # encode plotly graphs in JSON
    ids = ["figure-{}".format(i) for i, _ in enumerate(figures)]
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly figures
    return render_template('master.html', ids=ids, figuresJSON=figuresJSON, data_set=df)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[3:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template('go.html',
                           query=query,
                           classification_result=classification_results
                           )


def main():
    app.run(debug=True)


if __name__ == '__main__':
    main()
