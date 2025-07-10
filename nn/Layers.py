import numpy as np
from nn.Activations import Sigmoid,ReLU
from nn.Initializer import XavierUniform,HeUniform
class Dense:
    def __init__(self,units,activation,in_features):
        self.units = units
        self.in_features = in_features

# Tries to assign the initiliser for respective Activation,
# For now it is done automatically in future need to provide option for user to change
        if isinstance(activation,Sigmoid):
            self.init = HeUniform()
        else:
            self.init = XavierUniform()


        self.W = self.init(in_features,units)
        self.B = self.init(1,units)

        self.g = activation

    def __call__(self,X):
        if X.shape[1] != self.W.shape[0]:
            raise ValueError(f"Invalid Input dim {X.shape}, {self.W.shape}")

        Z = np.matmul(X,self.W) + self.B
        A = self.g(Z)

        return A

    def backward(self,A,y):
        return self.g.deriv(A,y)
