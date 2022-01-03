"""Data Wrangling"""
import pandas                           as pd
import numpy                            as np
from sqlalchemy                         import create_engine

"""Text Processing"""
import sys
import re
import pickle
import nltk
from nltk.tokenize                      import word_tokenize,sent_tokenize
from nltk.stem                          import WordNetLemmatizer
from nltk.corpus                        import stopwords
import                                  spacy

"""Machine Learning"""
from sklearn.metrics                    import confusion_matrix
from sklearn.model_selection            import train_test_split
from sklearn.linear_model               import LogisticRegression
from sklearn.feature_extraction.text    import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multioutput                import MultiOutputClassifier
from sklearn.pipeline                   import Pipeline
from sklearn.metrics                    import accuracy_score, precision_score, recall_score, f1_score
from sklearn.multioutput                import ClassifierChain
from sklearn.decomposition              import TruncatedSVD
import                                  joblib

"""Miscellaneous"""
from WordVectorTransformer import WordVectorTransformer

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath='./data/disasterResponse_db.db'):
    """Load message data from the database.

    Args:
        database_filepath (str, optional)   : Database of text messages. 
                                              Defaults to './data/DisasterResponse.db'.
    Returns:
        X, y (Dataframes)                   : The body of the message (X) and the predicted target (y). 
    """
    
    engine          = create_engine(f'sqlite:///{database_filepath}')
    df              = pd.read_sql_table('disasterResponse_db_table', engine)
    df.drop(columns=['child_alone'], axis=1, inplace=True)
    X               = df['message']
    Y               = df.iloc[:,5:]
    category_names  = Y.columns

    return X, Y, category_names

def tokenize(text):
    """Tokenization Function

    Args:
        text (String)   : List of text messages in english.

    Returns:
        clean_tokens    : Cleaned & lemmatized tokens.  
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """Constructs the pipeline with the neari optimal hyper-parameters found from GridSearchCV.

    Returns:
        pipeline (Pipeline Object : scikit-learn): Pipeline object.
    """
    pipeline = Pipeline([
        ('vect',      WordVectorTransformer()),
        ('trnc_svd',  TruncatedSVD(n_components=10, n_iter=5)),
        ('clf_chain', ClassifierChain(LogisticRegression(
            solver='lbfgs', random_state=0), order='random', random_state=42))
    ])
       
    return pipeline

def eval_metrics(ArrayL, ArrayP, col_names):

    """
    Github: https://github.com/atwahsz/Disaster-Response-Pipeline/blob/master/ML%20Pipeline%20Preparation.ipynb
    """

    """Evalute metrics of the ML pipeline model.

    Args:
        ArrayL (array)      : Array containing the ground truth.
        ArrayP (array)      : Array containing the predicted label.
        col_names (List)    : List of strings containing the target label names.

    Returns:
        metrics (DataFrame) : Contains accuracy, precision, recall 
                              and f1 score for a given set of ArrayL and ArrayP labels.
    """
    metrics = []

    # Evaluate metrics for each set of labels
    for i in range(len(col_names)):
        accuracy    = accuracy_score(ArrayL[:, i], ArrayP[:, i])
        precision   = precision_score(ArrayL[:, i], ArrayP[:, i], 
                                    average='weighted', labels=np.unique(ArrayP))
        recall      = recall_score(ArrayL[:, i], ArrayP[:, i], 
                                    average='weighted', labels=np.unique(ArrayP))
        f1          = f1_score(ArrayL[:, i], ArrayP[:, i],
                                    average='weighted', labels=np.unique(ArrayP))

        metrics.append([accuracy, precision, recall, f1])

    # store metrics
    metrics = np.array(metrics)
    data_metrics = pd.DataFrame(data=metrics, index=col_names, columns=[
                                'Accuracy', 'Precision', 'Recall', 'F1'])

    return data_metrics

def evaluate_model(model, X_test, Y_test, X_train, Y_train, category_names):
    """Performance calculation - Accuracy, Precission, Recall, F1_Score.

    Args:
        model (Pipeline Object : scikit-learn): Pipeline object.
        X_test          (DataFrame): Text messages  - 20% Split.
        Y_test          (DataFrame): Target labels  - 20% Split.
        X_train         (DataFrame): Text messages  - 80% Split.
        Y_train         (DataFrame): Text messages  - 20% Split.
        category_names  (List of Strings): Target names. 
    """
    
    y_train_pred    = model.predict(X_train)
    y_test_pred     = model.predict(X_test)

    train_performance = eval_metrics(
        np.array(Y_train), y_train_pred, category_names)

    test_performance = eval_metrics(
        np.array(Y_test), y_test_pred, category_names)

    relative_error = (
        np.abs((train_performance - test_performance))/test_performance)*100

    print('Performance on the Training Set.')
    print(train_performance)

    print('Performance on the Test Set.')
    print(test_performance)

    print('Relative difference in (%)')
    print(relative_error)

def save_model(model, model_filepath):
    """Save the pipeline as pickle object.

    Args:
        model (Pipeline Object : scikit-learn): Pipeline object.
        model_filepath (String): The filepath to save the pickle file.
    """

    # Save pipeline
    joblib.dump(model, model_filepath)

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
        evaluate_model(model, X_test, Y_test, X_train, Y_train, category_names)

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