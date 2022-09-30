class HyperParameters():
    
    @staticmethod
    def get_params(name):
        model = getattr(HyperParameters, name)
        return {key:value for key, value in model.__dict__.items() 
            if not key.startswith('__') and not callable(key)}
    
    class OrdinaryLeastSquares():
        pass # No hyperparamters to set
        
    class LeastMeanSquares():
        model_kwargs = dict(
            alpha = None, # TODO (REQUIRED) Set your learning rate
            epochs = None, # TODO (OPTIONAL) Set number of epochs
            seed = None, # TODO (OPTIONAL) Set seed for randomly generated weights
        )

        data_prep_kwargs = dict(
            # TODO (OPTIONAL) Set the names of the features/columns to use for the Housing dataset
            use_features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                            'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'],
        )

    class PolynomialRegression():
        model_kwargs = dict(
            degree = None, # TODO (REQUIRED) Set your polynomial degree
        )
        
    class PolynomialRegressionRegularized():
        model_kwargs = dict(
            degree = None, # TODO (REQUIRED) Set your polynomial degree
            lamb = None, # TODO (REQUIRED) Set your regularization value for lambda
        )

        data_prep_kwargs = dict(
            # TODO (OPTIONAL) Set the names of the features/columns to use for the Housing dataset
            use_features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                            'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'],
        )