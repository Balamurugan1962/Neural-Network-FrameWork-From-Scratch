import numpy as np

class MeanSquared:
    def __call__(self,A,y):
        return np.power(y-A,2)/2

    def deriv(self,A,y):
        return A-y

class BinaryCrossEntropy:
    def __call__(self,A,y):
        A = np.clip(A, 1e-15, 1 - 1e-15)
        loss = (-1 * y * np.log(A))-((1-y)*np.log(1-A))
        return (-1 * y * np.log(A))-((1-y)*np.log(1-A))

    def deriv(self,A,y):
        A = np.clip(A, 1e-15, 1 - 1e-15)
        return (-1*y/A) + ((1-y)/(1-A))

class CategoricalCrossEntropy:
    def __call__(self,A,y):
        loss = -np.sum(y * np.log(A + 1e-9))
        return loss


    def deriv(self,A,y):
        #For now it is only used with softmax on last layer
        # so We assume this loss is only used with Softmax Activation as a output activation
        return (A - y)
