import numpy as np

from itcs4156.models.LinearModel import LinearModel

class LeastMeanSquares(LinearModel):
    """
        Performs regression using least mean squares (gradient descent)
    
        attributes:
            w (np.ndarray): weight matrix
            
            alpha (float): learning rate or step size
            
            epochs (int): Number of epochs to run for mini-batch
                gradient descent
                
            seed (int): Seed to be used for NumPy's RandomState class
                or universal seed np.random.seed() function.
    """
    def __init__(self, alpha: float, epochs: int, seed: int = None):
        super().__init__()
        self.w = None
        self.alpha = alpha
        self.epochs = epochs
        self.seed = seed
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Used to train our model to learn optimal weights.
        
            TODO:
                Finish this method by adding code to perform LMS in order to learn the 
                weights `self.w`.
        """
        pass # TODO replace this line with your code
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Used to make a prediction using the learned weights.
        
            TODO:
                Finish this method by adding code to make a prediction given the learned
                weights `self.w`.
        """
        # TODO (REQUIRED) Add code below

        # TODO (REQUIRED) Store predictions below by replacing np.ones()
        y_hat = np.ones([len(X), 1])
        
        return y_hat
