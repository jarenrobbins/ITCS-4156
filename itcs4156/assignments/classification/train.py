from pdb import set_trace
from tkinter import Image
import numpy as np
from sklearn.pipeline import Pipeline

from itcs4156.util.data import AddBias, Standardization, ImageNormalization, OneHotEncoding

class HyperParametersAndTransforms():
    
    @staticmethod
    def get_params(name):
        model = getattr(HyperParametersAndTransforms, name)
        params = {}
        for key, value in model.__dict__.items():
            if not key.startswith('__') and not callable(key):
                if not callable(value) and not isinstance(value, staticmethod):
                    params[key] = value
        return params
    
    class Perceptron():
        """Kwargs for classifier the Perceptron class and data prep"""
        model_kwargs = dict(
            alpha = 0.1,  # (REQUIRED) Set learning rate 
            epochs = 1,  # (REQUIRED) Set epochs
            seed = None, # (OPTIONAL) Set seed for reproducible results
        )

        data_prep_kwargs = dict(
            # (OPTIONAL) Add Pipeline() definitions below
            target_pipe = None,
            # (REQUIRED) Add Pipeline() definitions below
            feature_pipe = Pipeline([
                ('std', Standardization()),
                ('bias', AddBias()),
            ])
        )
        
    class NaiveBayes():
        """Kwargs for classifier the NaiveBayes class and data prep"""
        model_kwargs = dict(
            smoothing = 10e-2, # (OPTIONAL) Set smoothing parameter for STD
        )
        
        data_prep_kwargs = dict(
            # (OPTIONAL) Add Pipeline() definitions below
            target_pipe = None,
            # (REQUIRED) Add Pipeline() definitions below
            feature_pipe = Pipeline([
                ('norm', ImageNormalization())
            ])
        )
        
    class LogisticRegression():
        model_kwargs = dict(
            alpha = .05, # (REQUIRED) Set learning rate
            epochs = 100, # (REQUIRED) Set epochs
            seed = None, # (OPTIONAL) Set seed for reproducible results
            batch_size = None, # (OPTIONAL) Set mini-batch size if using mini-batch gradient descent
        )
      
        data_prep_kwargs = dict(
            # (REQUIRED) Add Pipeline() definitions below
            target_pipe = Pipeline([
                ('ohe', OneHotEncoding())
            ]),
            # (REQUIRED) Add Pipeline() definitions below
            feature_pipe = Pipeline([
                ('norm', Standardization()),
                ('bias', AddBias())
            ]) 
        )