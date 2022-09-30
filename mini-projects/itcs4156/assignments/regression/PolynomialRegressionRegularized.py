import numpy as np

from itcs4156.assignments.regression.PolynomialRegression import PolynomialRegression

class PolynomialRegressionRegularized(PolynomialRegression):
    """
        Performs polynomial regression with l2 regularization using the ordinary least squares algorithm
    
        attributes:
            w (np.ndarray): weight matrix that is inherited from OrdinaryLeastSquares
            
            degree (int): the number of polynomial degrees to include when adding
                polynomial features.
    """

    def __init__(self, degree: int, lamb: float):
        super().__init__(degree)
        self.lamb = lamb
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Used to train our model to learn optimal weights.
        
            TODO:
                Finish this method by adding code to perform polynomial regression using
                the closed form solution OLS with l2 regularization to learn 
                the weights `self.w`.
                
            Hint:
                Add the bias after computing polynomial features. Typically we don't want
                to include the bias when computing polynomial features.
        """
        pass # TODO replace this line with your code