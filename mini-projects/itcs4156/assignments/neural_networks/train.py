import numpy as np
from sklearn.pipeline import Pipeline
from itcs4156.util.metrics import nll, sse
from itcs4156.util.data import AddBias, Standardization, ImageNormalization, OneHotEncoding

def delta_mse(y, y_hat):
    # (REQUIRED) Add code below for the derivative of the mean squared error
    return y_hat - y

def delta_softmax_nll(y, y_hat):
    # (REQUIRED) Add code below for for the combined derivative of the softmax and negative log likelihood loss
    return y_hat - y

class Linear():
    @staticmethod
    def activation(z):
        # (REQUIRED) Add code below for Linear activation function equation
        return z
    
    @staticmethod
    def derivative(z):
        # (REQUIRED) Add code below for Linear activation function derivative
        return np.ones(z.shape)
    
class Sigmoid():
    @staticmethod
    def activation(z):
        # (REQUIRED) Add code below for Sigmoid activation function equation
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def derivative(z):
        # (REQUIRED) Add code below for Sigmoid activation function derivative
        return Sigmoid.activation(z) * (1 -  Sigmoid.activation(z))

class Tanh():
    @staticmethod
    def activation(z):
        # (REQUIRED) Add code below for Tanh activation function equation
        return np.tanh(z)
    
    @staticmethod
    def derivative(z):
        # (REQUIRED) Add code below for Tanh activation function derivative
        return 1 - np.tanh(z)**2

class ReLU():
    @staticmethod
    def activation(z):
         # (REQUIRED) Add code below for ReLU activation function equation
        return np.maximum(0, z)
    
    @staticmethod
    def derivative(z):
        # (REQUIRED) Add code below for ReLU activation function derivative
        z = z.copy()
        z[z>=0] = 1
        z[z<0] = 0
        return z

class Softmax():
    @staticmethod
    def activation(z):
        # (REQUIRED) Add code below for softmax activation function equation
        z = z - np.max(z, axis=0, keepdims=True)
        e_z = np.exp(z)
        denominator = np.sum(e_z, axis=0, keepdims=True)

        return e_z / denominator
    
    @staticmethod
    def derivative(z):
        # (REQUIRED) Add code below for softmax activation function derivative
        # placeholder derivative
        # THIS IS NOT THE ACTUAL DERIVATIVE OF THE SOFTMAX
        return np.ones(z.shape)

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
    
    class NeuralNetworkRegressor():
        """Kwargs for regression neural network and data prep"""
        model_kwargs = dict(
            neurons_per_layer = [50, 50, 1], # TODO (REQUIRED) Set neural network neurons per layer
            learning_curve_loss = sse,
            delta_loss_func = delta_mse, # TODO (REQUIRED) Set neural network's loss function derivative
            g_hidden = Sigmoid, # TODO (REQUIRED) Set neural network's hidden neurons activation function
            g_output = Linear,  # TODO (REQUIRED) Set neural network's output neurons activation function
            alpha = .01, # TODO (REQUIRED) Set neural network's learning rate
            epochs = 500,  # TODO (REQUIRED) Set neural network's  epochs
            batch_size = 32, # TODO (REQUIRED) Set neural network's mini-batch size
            verbose = False, # TODO (OPTIONAL) Set to allow neural network to print debugging statements during training 
            seed = None, # TODO (OPTIONAL) Set the neural network to random state seed 
        )
        
        # (OPTIONAL) model kwargs used for hyper-parameter search.
        # EVERY argument must be wrapped in a list.
        search_model_kwargs = dict(
            neurons_per_layer = [[]], # TODO (OPTIONAL) Set neural network neurons per layer
            learning_curve_loss = [sse],
            delta_loss_func = [None], # TODO (OPTIONAL) Set neural network's loss function derivative
            g_hidden = [None], # TODO (OPTIONAL) Set neural network's hidden neurons activation function
            g_output = [None],  # TODO (OPTIONAL) Set neural network's output neurons activation function
            alpha = [None], # TODO (OPTIONAL) Set neural network's learning rate
            epochs = [1],  # TODO (OPTIONAL) Set neural network's  epochs
            batch_size = [32], # TODO (OPTIONAL) Set neural network's mini-batch size
            verbose = [False], # TODO (OPTIONAL) Set to allow neural network to print debugging statements during training 
            seed = [None], # TODO (OPTIONAL) Set the neural network to random state seed 
        )
        
        data_prep_kwargs = dict(
            # (OPTIONAL) Add Pipeline() definitions below
            target_pipe = None,
            # (REQUIRED) Add Pipeline() definitions below
            feature_pipe = Pipeline([
                ('scaler', Standardization())
            ]),
            # (OPTIONAL) Set the names of the features/columns to use for the Housing dataset
            use_features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                            'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'],
        )
        
    class NeuralNetworkClassifier():
        """Kwargs for classifier neural network and data prep"""
        model_kwargs = dict(
            neurons_per_layer = [30, 10], # TODO (REQUIRED) Set neural network neurons per layer
            learning_curve_loss = nll,
            delta_loss_func = delta_softmax_nll, # TODO (REQUIRED) Set neural network's loss function derivative
            g_hidden = ReLU, # TODO (REQUIRED) Set neural network's hidden neurons activation function
            g_output = Softmax,  # TODO (REQUIRED) Set neural network's output neurons activation function
            alpha = .01, # TODO (REQUIRED) Set neural network's learning rate
            epochs = 500,  # TODO (REQUIRED) Set neural network's  epochs
            batch_size = 64, # TODO (REQUIRED) Set neural network's mini-batch size
            verbose = False, # TODO (OPTIONAL) Set to allow neural network to print debugging statements during training 
            seed = None, # TODO (OPTIONAL) Set the neural network to random state seed 
        )
        
        # (OPTIONAL) model kwargs used for hyper-parameter search.
        # EVERY argument must be wrapped in a list.
        search_model_kwargs = dict(
            neurons_per_layer = [[]], # TODO (OPTIONAL) Set neural network neurons per layer
            learning_curve_loss = [nll],
            delta_loss_func = [None], # TODO (OPTIONAL) Set neural network's loss function derivative
            g_hidden = [None], # TODO (OPTIONAL) Set neural network's hidden neurons activation function
            g_output = [None],  # TODO (OPTIONAL) Set neural network's output neurons activation function
            alpha = [None], # TODO (OPTIONAL) Set neural network's learning rate
            epochs = [1],  # TODO (OPTIONAL) Set neural network's  epochs
            batch_size = [32], # TODO (OPTIONAL) Set neural network's mini-batch size
            verbose = [False], # TODO (OPTIONAL) Set to allow neural network to print debugging statements during training 
            seed = [None], # TODO (OPTIONAL) Set the neural network to random state seed 
        )
        
        data_prep_kwargs = dict(
            # TODO (REQUIRED) Add Pipeline() definitions below
            target_pipe = Pipeline([
                ('one-hot', OneHotEncoding())
            ]),
            # TODO (REQUIRED) Add Pipeline() definitions below
            feature_pipe = Pipeline([
                ('standard', Standardization())
            ]),
        )