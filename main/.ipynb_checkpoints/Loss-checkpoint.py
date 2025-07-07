import numpy as np


class BinaryCrossEntropy:
    def __call__(self,A,y):
        return (-1 * y * np.log(A))-((1-y)*np.log(1-A))

    def deriv(self,A,y):
        return (-1*y/A) + ((1-y)/(1-A))

class SparseCategoricalCrossEntropy:
    def __call__(self,A,y):
        L = np.zeros_like(A)
        L[0][y] = -1* np.log(A[0][y])
        print(L,y,A[0][y])
        return L


    def deriv(self,A,y):
        #For now it is only used with softmax on last layer
        # so We assume this loss is only used with Softmax Activation as a output activation
        return 1
