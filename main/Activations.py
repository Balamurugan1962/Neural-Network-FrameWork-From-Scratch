import numpy as np

class Linear:
    def __call__(self,A):
        return A

    def deriv(self,A,y):
        g = np.ones_like(A)
        return g

class Sigmoid:
    def __call__(self,A):
        return 1 / (1+np.exp(-1 * A))

    def deriv(self,A,y):
        return A * (1-A)

class ReLU:
    def __call__(self,A):
        return np.maximum(0, A)

    def deriv(self,A,y):
        return A>0

class Softmax:

    def __call__(self,A):
        exps = np.exp(A - np.max(A, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    #considered to be used with Categorical Loss function
    def deriv(self,A,y):
        return 1
