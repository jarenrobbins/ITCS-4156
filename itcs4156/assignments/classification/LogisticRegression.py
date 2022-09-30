from typing import List, Tuple, Union 

import numpy as np

from itcs4156.models.ClassificationModel import ClassificationModel

class LogisticRegression(ClassificationModel):
    """
        Performs Logistic Regression using the softmax function.
    
        attributes:
            alpha: learning rate or step size used by gradient descent.
                
            epochs: Number of times data is used to update the weights `self.w`.
                Each epoch means a data sample was used to update the weights at least
                once.
            
            seed (int): Seed to be used for NumPy's RandomState class
                or universal seed np.random.seed() function.
            
            batch_size: Mini-batch size used to determine the size of mini-batches
                if mini-batch gradient descent is used.
            
            w (np.ndarray): NumPy array which stores the learned weights.
    """
    def __init__(self, alpha: float, epochs: int = 1,  seed: int = None, batch_size: int = None):
        ClassificationModel.__init__(self)
        self.alpha = alpha
        self.epochs = epochs
        self.seed = seed
        self.batch_size = batch_size
        self.w = None

    def softmax(self, z: np.ndarray) -> np.ndarray:
        """ Computes probabilities for multi-class classification given continuous inputs z.
        
            Args:
                z: Continuous outputs after dotting the data with the current weights 

            Finish this method by adding code to return the softmax. Don't forget
            to subtract the max from `z` to maintain  numerical stability!
        """
        z = z - np.max(z, axis=-1, keepdims=True)
        e_z = np.exp(z)
        denominator = np.sum(e_z, axis=-1, keepdims=True)

        return e_z / denominator

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Train our model to learn optimal weights for classifying data.
        
            Args:
                X: Data 
                
                y: Targets/labels
                
            Finish this method by using either batch or mini-batch gradient descent
            to learn the best weights to classify the data. You'll need to finish and 
            also call the `softmax()` method to complete this method. Also, update 
            and store the learned weights into `self.w`. 
        """
        rng = np.random.RandomState(self.seed)
        self.w = rng.rand(X.shape[1], 1)
        for e in range(self.epochs):
            z = X @ self.w

            probs = self.softmax(z)
            avg_gradient = (X.T @ (probs - y)) / len(y)
            self.w = self.w - self.alpha * avg_gradient

       
    def predict(self, X: np.ndarray):
        """ Make predictions using the learned weights.
        
            Args:
                X: Data 

            Finish this method by adding code to make a prediction given the learned
            weights `self.w`. Store the predicted labels into `y_hat`.
        """
        # Add code below
        z = X @ self.w
        probs = self.softmax(z)
        y_hat = np.argmax(probs, axis=1)  # Store predictions here by replacing np.ones()
        # Makes sure predictions are given as a 2D array
        return y_hat.reshape(-1, 1)
