import re
import sys
import spacy
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, fbeta_score, make_scorer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import warnings
warnings.filterwarnings('ignore')
nlp = spacy.load('en_core_web_sm')


def load_data(database_filepath):
    """
    Loads data from SQL Database

    Args:
       database_filepath: SQL database file
    Returns:
       x: Features dataframe
       y: Target dataframe
       labels: Target labels
    """

    # Load database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('database', con=engine)

    # Separate into features and target
    x = df['message'].values
    y = df.iloc[:, 3:].values
    labels = df.iloc[:, 3:].columns

    return x, y, labels


def tokenise(row):
    """
    Tokenises text data

    Args:
        row: row of dataframe
    Returns:
        clean_row: Processed text after normalising, tokenising and lemmatising
    """

    # Creating doc object for SpaCy
    doc = nlp(re.sub(r"[^a-zA-Z0-9]", ' ', row.lower()))
    clean_row = []

    for tok in doc:
        clean_tok = tok.lemma_.strip()  # Lemmatisation
        if len(clean_tok) > 1 and clean_tok != '-PRON-' and not tok.is_stop:
            clean_row.append(clean_tok)

    return clean_row


def build_model():
    """
    Build model with GridSearchCV

    Returns:
        Trained model after performing grid search
    """

    # model pipeline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenise)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))])

    # pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenise)),
    #                      ('tfidf', TfidfTransformer()),
    #                      ('clf',
    #                       MultiOutputClassifier(RandomForestClassifier(n_estimators=50, n_jobs=-1,
    #                                                                    random_state=69)))])

    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                  'vect__max_df': (0.75, 1.0)}

    # Use f_beta score to optimise for recall
    model = GridSearchCV(pipeline, parameters, n_jobs=-1, cv=2, verbose=3)

    return model


def train(x, y, model, labels):
    # train test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=69)

    # fit model
    model.fit(x_train, y_train)

    # output model test results
    y_pred = model.predict(x_test)

    # Getting classification report
    print(classification_report(y_test, y_pred, target_names=labels))

    # Calculating accuracy score
    print('Accuracy: {}'.format(np.mean(y_test == y_pred)))

    return model


def save_model(model, model_filepath):
    # Export model as a pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        x, y, labels = load_data(database_filepath)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model = train(x, y, model, labels)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database as the first argument and the filepath of '
              'the pickle file to save the model to as the second argument. \n\nExample: '
              'python train_classifier.py ../data/database.db model_svc.pkl')


if __name__ == '__main__':
    main()
