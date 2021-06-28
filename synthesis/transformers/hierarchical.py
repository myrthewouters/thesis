import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class GeneralizeSchematic(TransformerMixin, BaseEstimator):
    
    def __init__(self, schema_dict):
        self.schema_dict = schema_dict

    def fit(self, col=None):
        return self

    def transform(self, col):
        """Replaces all values in col with its generalized form in schema dict"""

        to_check = list(self.schema_dict.keys())
        to_check += ['nan', np.nan] # Allow nan values in column

        # Check if all values in column are also in schema dict as keys
        assert all([val in to_check for val in col]), "Column contains values not in schema dict"
        
        return col.replace(self.schema_dict)