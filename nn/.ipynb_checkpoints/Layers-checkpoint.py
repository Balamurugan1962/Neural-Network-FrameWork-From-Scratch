import numpy as np

class Dense:
    def __init__(self,units,activation,in_features):
        self.units = units
        self.in_features = in_features

        #Xavier Initilisation
        limit = np.sqrt(6 / (in_features + units))
        self.W = np.random.uniform(-limit, limit, (in_features, units))
        limit = np.sqrt(6 / (1 + units))
        self.B = np.random.uniform(-limit, limit, (1, units))

        self.g = activation

    def __call__(self,X):
        if X.shape[1] != self.W.shape[0]:
            raise ValueError(f"Invalid Input dim {X.shape}, {self.W.shape}")

        Z = np.matmul(X,self.W) + self.B
        A = self.g(Z)

        return A

    def backward(self,A,y):
        return self.g.deriv(A,y)
