import numpy as np

from itcs4156.models.ClassificationModel import ClassificationModel

class Perceptron(ClassificationModel):
    """
        Performs Gaussian Naive Bayes
    
        attributes:
            alpha: learning rate or step size used by gradient descent.
                
            epochs (int): Number of times data is used to update the weights `self.w`.
                Each epoch means a data sample was used to update the weights at least
                once.
                
            seed (int): Seed to be used for NumPy's RandomState class
                or universal seed np.random.seed() function.
            
            batch_size (int): Mini-batch size used to determine the size of mini-batches
                if mini-batch gradient descent is used.
            
            w (np.ndarray): NumPy array which stores the learned weights.
    """
    def __init__(self, alpha: float, epochs: int = 1, seed: int = None):
        ClassificationModel.__init__(self)
        self.alpha = alpha
        self.epochs = epochs
        self.seed = seed
        self.w = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """ Train model to learn optimal weights when performing binary classification.
        
            Args:
                X: Data 
                
                y: Targets/labels
                
            Finish this method by using Rosenblatt's Perceptron algorithm to learn
            the best weights to classify the binary data. There is no need to
            implement th pocket algorithm unless you choose to do so. Also, update 
            and store the learned weights into `self.w`.
        """
        m_samples = X.shape[0]
        n_features = X.shape[1]
        
        # Randomly initialize weights 
        rng = np.random.RandomState(42)
        self.w = rng.rand(n_features)
        
        # Loop over dataset multiple times
        for e in range(self.epochs):
            misclassified = 0
            # Loop over all samples
            for i in range(m_samples):
                # Compute continuous prediction z
                z = self.w @ X[i]
                
                # Computes discrete prediction by applying
                # the sign activation function.
                # Alternatively you could just call self.predict(X[i])
                y_hat = np.sign(z)
                
                # Check if data sample was misclassified
                if y_hat != y[i]:
                    # Update rule
                    self.w = self.w + self.alpha * y[i] * X[i]
                    misclassified += 1
            # Check convergence
            if misclassified == 0:
                return
       
   
    def predict(self, X: np.ndarray):
        """ Make predictions using the learned weights.
        
            Args:
                X: Data 

            Finish this method by adding code to make a prediction given the learned
            weights `self.w`. Store the predicted labels into `y_hat`.
        """
        # Add code below
        z = X @ self.w
        y_hat = np.sign(z) # Store predictions here by replacing np.ones()
        # Makes sure predictions are given as a 2D array
        return y_hat.reshape(-1, 1)

