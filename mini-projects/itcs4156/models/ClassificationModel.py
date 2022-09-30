from abc import abstractmethod

import numpy as np

from itcs4156.models.BaseModel import BaseModel

class ClassificationModel(BaseModel):
    """
        Abstract class for classification 
        
        Attributes
        ==========
    """

    # check if the matrix is 2-dimensional. if not, raise an exception    
    def _check_matrix(self, mat, name):
        if len(mat.shape) != 2:
            raise ValueError(f"Your matrix {name} shape is not 2D! Matrix {name} has the shape {mat.shape}")
        
    ####################################################
    #### abstract funcitons ############################
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
            train classification model
            
            Args:
                X:  Input data
                
                y:  targets/labels
        """        
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray):
        """
            apply the learned model to input X
            
            parameters
            ----------
            X     2d array
                  input data
            
        """        
        pass 