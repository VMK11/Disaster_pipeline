"""Machine Learning"""
from sklearn.base import BaseEstimator, TransformerMixin
import            spacy

"""Data Wrangling"""
import numpy                            as np


class WordVectorTransformer(TransformerMixin,BaseEstimator):
    def __init__(self, model="en_core_web_lg"):
        self.model = model

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        nlp = spacy.load(self.model)
        return np.concatenate([nlp(doc).vector.reshape(1,-1) for doc in X])