import numpy as np


class Sigmoid:
    def __call__(self,A):
        g = A.copy()
        g[0] = 1 / (1+np.exp(-1 * A[0]))
        return g

    def deriv(self,A,y):
        return A * (1-A)

class ReLU:
    def __call__(self,A):
        g = np.zeros_like(A)
        g[g<0] = 0
        return g

    def deriv(self,A,y):
        return A>0

class Softmax:
    def __call__(self,A):
        g = A.copy()
        exps = np.exp(A[0])
        g[0] = exps/np.sum(exps)
        return g

    #considered to be used with Categorical Loss function
    def deriv(self,A,y):
        grad = A.copy()
        grad[0][y] -=1;
        return grad
