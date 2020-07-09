import sys
import os
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine
import pickle

#for cleaning the text
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#for building model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

#for evluation
from sklearn.metrics import classification_report, accuracy_score, precision_score
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    #table_name = os.path.basename(database_filepath).replace(".db","").lower()
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.iloc[:, 5:]
    category_names = list(df.columns[5:])

    return X, Y, category_names


def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    # tokenize text
    tokens = nltk.word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                    ])

    # parameters = {'clf__estimator__max_depth':[None, 20, 40], 
    # #'clf__estimator__min_samples_split': [2, 4]
    #              }
    # cv = GridSearchCV(pipeline, param_grid=parameters)

    #pipeline.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Input:
    Output:

    Notes: Given the context of this dataset, it is important to look at the precison score. We don't want to miss a message that actually reports child-alone after a disaster. [/end]
    """
    Y_pred = model.predict(X_test)

    precision = []

    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i])))
        
        #precision score
        test = Y_test.iloc[:, i]
        test_numpy = np.array(test.values.tolist())
        precision.append(precision_score(test_numpy, Y_pred[:, i], average='weighted'))
        print('Precision Socre is: %.2f' %(np.mean(precision)))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()