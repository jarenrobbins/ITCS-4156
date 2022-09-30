from typing import List, Tuple, Union, Callable

import numpy as np
from sklearn.base import BaseEstimator

class Layer():
    """ Class which stores all variables required for a layer in a neural network
    
        Attributes:
            W: NumPy array of weights for all neurons in the layer
            
            b: NumPy array of biases for all neurons in the layer
            
            g: Activation function for all neurons in the layer
            
            name: Name of the layer
            
            neurons: Number of neurons in the layer
            
            inputs: Number of inputs into the layer
            
            Z: Linear combination of weights and inputs for all neurons. 
                Initialized to an empty array until it is computed and set.
                
            A: Activation output for all neurons. Initialized to an empty 
                array until it is computed and set.
    """
    def __init__(
        self, 
        W:np.array, 
        b:np.array, 
        g: object, 
        name: str=""
    ):
        self.W = W
        self.b = b
        self.g = g
        self.name = name 
        self.neurons = len(W)
        self.inputs = W.shape[1]
        self.Z = np.array([])
        self.A = np.array([])
    
    def print_info(self) -> None:
        """ Prints info for all class attributes"""
        print(f"{self.name}")
        print(f"\tNeurons: {self.neurons}")
        print(f"\tInputs: {self.inputs}")
        print(f"\tWeight shape: {self.W.shape}")
        print(f"\tBias shape: {self.b.shape}")
        print(f"\tActivation function: {self.g.__name__}")
        print(f"\tZ shape: {self.Z.shape}")
        print(f"\tA shape: {self.A.shape}")

def get_mini_batches(data_len: int, batch_size: int = 32) -> List[np.ndarray]:
    """ Generates mini-batches based on the data indexes
        
        Args:
            data_len: Length of the data
            
            batch_size: Size of each mini batch where the last mini-batch
                might be smaller than the rest if the batch_size does not 
                evenly divide the data length.
    
    """
    X_idx = np.arange(data_len)
    np.random.shuffle(X_idx)
    batches = [X_idx[i:i+batch_size] for i in range(0, data_len, batch_size)]
    
    return batches

class NeuralNetwork(BaseEstimator):
    """ Runs the initialization and training process for a multi-layer neural network.
        
        Attributes:
            neurons_per_layer: A list where each element represents 
                    the neurons in a layer. For example, [2, 3] would
                    create a 2 layer neural network where the hidden layer
                    has 2 neurons and the output layer has 3 neurons.
            
            learning_curve_loss: Pointer to a function which computes the SSE or NLL loss.
                This loss will be tracked for each mini-batch and epoch. The loss computed
                will be stored in the avg_trn_loss_tracker and avg_vld_loss_tracker variables.

            delta_loss_func: Pointer to a function which computes the  derivative for
                the MSE or NLL loss.

            g_hidden: Activation function used by ALL neurons 
                in ALL hidden layers.
                    
            g_output: Activation function used by ALL neurons
                in the output layer.
        
            alpha: learning rate or step size used by gradient descent.
                
            epochs: Number of times data is used to update the weights `self.w`.
                Each epoch means a data sample was used to update the weights at least
                once.
            
            batch_size: Mini-batch size used to determine the size of mini-batches
                if mini-batch gradient descent is used.
            
            seed: Random seed to use when initializing the layers of the neural network.

            verbose: If True, print statements inside the train() method will
                be printed.

            nn: A list of Layer class instances which define the neural network.

            avg_trn_loss_tracker: A list that tracks the average training loss per epoch. 

            avg_vld_loss_tracker: A list that tracks the average validation loss per epoch.
            
    """
    def __init__(
        self,
        neurons_per_layer: List[int],
        learning_curve_loss: Callable,
        delta_loss_func: Callable,
        g_hidden: object,
        g_output: object,
        alpha: float = .001, 
        epochs: int = 1, 
        batch_size: int = 64,
        seed: int = None,
        verbose: bool = False,
    ):
        self.neurons_per_layer = neurons_per_layer
        self.learning_curve_loss = learning_curve_loss
        self.delta_loss_func = delta_loss_func
        self.g_hidden = g_hidden
        self.g_output = g_output
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose

        self.nn = []
        self.avg_trn_loss_tracker = []
        self.avg_vld_loss_tracker = []

    def init_neural_network(self, n_input_features: int)-> List[Layer]:
        """ Initializes weights and biases for a multi-layer neural network 
        
            Args:
                n_input_features: Number of features the input data has
                
            Finish this method by completing the for loop to initialize the weights
            `W` and biases `b`. Once initialized, create an instance of the `Layer`
            class by passing the required arguments of weights `W`, biases `b`, 
            activation function `g`, and name `name` and then append it to the 
            `nn` list. Return the completed neural network `nn` once the for-loop
            has finished.

        """
        nn = []
        # Set numpy global random seed
        np.random.seed(self.seed)
        for l, neurons in enumerate(self.neurons_per_layer):
            # Set inputs to number of input features
            # for the first hidden layer
            if l == 0:
                inputs = n_input_features
            else:
                inputs = self.neurons_per_layer[l-1]
            
            # Set activation functions for the output
            # layer neurons and set the names of the nn
            if l == len(self.neurons_per_layer)-1:
                g = self.g_output
                name = f"Layer {l+1}: Output Layer"
            else:
                g = self.g_hidden
                name = f"Layer {l+1}: Hidden Layer"
            
            # (REQUIRED) Add code below
            W = NeuralNetwork.init_weights(self, neurons, inputs)
            
            b = np.ones([neurons, 1])
            
            layer = Layer(W=W, b=b, g=g, name=name)
            nn.append(layer)

        return nn

    def init_weights(self, neurons: int, inputs: int) -> np.ndarray:
        """ Initializes weight values
        
            Args:
                neurons: Number of neurons in the layer
                
                inputs: Number of inputs to the layer
            
            Finish this method by returning randomly initalized weights given
            the arguments for the number of neurons and inputs. Return the randomly
            initialized weights once done.
        """
        return np.random.uniform(low=-0.5, high=0.5, size=(neurons, inputs))
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        X_vld: np.ndarray = None, 
        y_vld: np.ndarray = None,
    ) -> None:
        """ Initializes and trains the defined neural network using gradient descent  
        
            Args:
                X: Training features/data 
                
                y: Training targets/labels

                X_vld: validation features/data which are used for computing the validation
                    loss after every epoch.

                y_vld: validation targets/labels which are used for computing the validation
                    loss after every epoch.
                    
            Finish this method by completing the training loop which performs 
            mini-batch gradient descent and tracks the training loss and validation
            scores per each epoch. To complete the training loop, you will need to
            initialize the neural network list `nn`, call the forward pass, and call
            the backwards pass.
        """
        m = len(X)
        self.avg_trn_loss_tracker = []
        self.avg_vld_loss_tracker = []
        
        # (REQUIRED) Initialize self.nn below by replacing []
        self.nn = NeuralNetwork.init_neural_network(
            self,
            n_input_features=X.shape[1]
        )
        
        for e in range(self.epochs):
            if self.verbose: print(f"Epoch: {e+1}")
            batches = get_mini_batches(data_len=m, batch_size=self.batch_size)
            total_trn_batch_loss = 0
            for mb in batches:
                # Forward pass to get predictions
                # (REQUIRED) Store the training forward pass predictions below by replacing np.zeros()
                y_hat = NeuralNetwork.forward(self, X[mb])

                # Backward pass to get gradients
                # (REQUIRED) Add backwards pass call below
                NeuralNetwork.backward(
                    self,
                    X=X[mb], 
                    y=y[mb], 
                    y_hat=y_hat
                )

                trn_batch_loss = self.learning_curve_loss(y[mb], y_hat)
                total_trn_batch_loss += trn_batch_loss
                
            avg_trn_loss = total_trn_batch_loss / m
            if self.verbose: print(f"\tTraining loss: {avg_trn_loss}")
            self.avg_trn_loss_tracker.append(avg_trn_loss)
            
            if X_vld is not None and y_vld is not None:
                m_vld = len(y_vld)
                # (REQUIRED) Store the validation forward pass predictions below by replacing np.zeros()
                y_hat_vld = NeuralNetwork.forward(self, X=X_vld)
                
                avg_vld_loss = self.learning_curve_loss(y_vld, y_hat_vld) / m_vld
                if self.verbose: print(f"\tValidation loss: {avg_vld_loss}")
                self.avg_vld_loss_tracker.append(avg_vld_loss)
            
    def forward(self, X:np.ndarray) -> np.ndarray:
        """ Performs the forward pass for a multi-layer neural network
    
            Args:
                X: Input features. This should be typically be the 
                    training data.
                     
            Finish this method by performing the forward pass for a multi-layer
            neural network. Return the output `y_hat` once done.
        """
        A = X.T
        
        # layer: the current Layer class instance
        # l: the index position of layer in the nn list
        for l, layer in enumerate(self.nn):
            
            layer.Z = layer.W @ A + layer.b
            if self.verbose:  print(f"Z{l+1}{layer.Z.shape} = W{l+1}{layer.W.shape} @ A{l}{A.shape} + b{l+1}{layer.b.shape}")
            
            layer.A = layer.g.activation(layer.Z)
            if self.verbose: print(f"A{l+1}{layer.A.shape} = g(Z{l+1}{layer.Z.shape})")
            
            A = layer.A
            
        y_hat = A.T
        return y_hat
    
    def backward(self, X:np.ndarray, y:np.ndarray, y_hat:np.ndarray) -> None:
        """ Performs the feedback process for a multi-layer neural network
        
            Args:
                X: Training features/data
                
                y: Training targets/labels
                
                y_hat: Training predictions (predicted targets or probabilities)

            Finish this method by performing the backward pass for a multi-layer
            neural network.
        """
        # Get index of nn and reverse the order
        # to start with output layer.
        layer_index = np.arange(len(self.nn))[::-1]
        
        delta_A = (y_hat - y).T

        for l, layer in zip(layer_index, self.nn[::-1]):
            # Get inputs from previous layer or grab
            # feature inputs for 1st hidden layer.
            if l == 0:
                A = X.T
            else:
                prev_layer = self.nn[l-1]
                A = prev_layer.A
                
            if self.verbose:
                print(f"{layer.name}")
                print(f"\tInputs: A{l}{A.shape}")
                
            delta_Z = delta_A * layer.g.derivative(layer.Z)
            if self.verbose: print(f"\tdelta_Z{l+1}{delta_Z.shape} = delta_A{l+1}{delta_A.shape} * g'(Z{l+1}{layer.Z.shape})")
                
            delta_W = delta_Z @ A.T
            if self.verbose: print(f"\tdelta_W{l+1}{delta_W.shape} = delta_Z{l+1}{delta_Z.shape} @ A{l}^T{A.T.shape}")
            
            W_avg_grad = delta_W / len(y)
            
            delta_b = delta_Z @ np.ones([1, len(y)]).T
            if self.verbose: print(f"\tdelta_b{l+1}{delta_b.shape} = sum(delta_Z{l+1}{delta_Z.shape})")

            b_avg_grad = delta_b / len(y) # np.mean(delta_b, axis=1, keepdims=True)
            if self.verbose: print(f"\tb{l+1} average gradient shape: {b_avg_grad.shape}")
            
            delta_A = layer.W.T @ delta_Z
            if self.verbose: print(f"\tdelta_A{l}{delta_A.shape} = W{l+1}^T{layer.W.T.shape} @ Z{l+1}{delta_Z.shape} ")
            
            layer.W = layer.W - self.alpha * W_avg_grad
            
            layer.b = layer.b - self.alpha * b_avg_grad