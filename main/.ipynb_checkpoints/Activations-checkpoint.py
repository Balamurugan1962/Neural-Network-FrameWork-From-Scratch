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
        g = A.copy()
        exps = np.exp(A[0])
        g[0] = exps/np.sum(exps)
        return g

    #considered to be used with Categorical Loss function
    def deriv(self,A,y):
        grad = A.copy()
        grad[0, y[0].item()] -= 1
        return grad
