import numpy as np
from main.Activations import Softmax
class Model:
    def __init__(self,layers):
        self.layers = layers



    def compile(self,loss,optimizer):
        self.loss = loss
        self.optimizer = optimizer



    def forward(self,A0):
        A_out = A0

        for layer in self.layers:
            A_out = layer(A_out)
        return A_out



    def backward(self,A0,y):

        l = len(self.layers)
        A = [0]*(l+1)

        A[0] = A0
        for i in range(1,l+1):
            A[i] = self.layers[i-1](A[i-1])
        dA = self.loss.deriv(A[l],y)

        dW = [0] * l
        dB = [0] * l

        for i in range(l,0,-1):
            dZ = dA * self.layers[i-1].backward(A[i],y)

            dW[i-1] = A[i-1].T @ dZ
            dB[i-1] = dZ.sum(axis=0,keepdims=True)

            dA = dZ @ (self.layers[i-1].W.T)

        return dW,dB


    def fit(self,X,y,epoch=100):
        m = X.shape[0]
        l = len(self.layers)
        for e in range(epoch):
            if (e+1) % 10 == 0:
                y_hat = self.forward(X[np.newaxis,0,:])
                print(f"epoch {e+1:>5}  loss = {self.loss(y_hat, y[0])}")

            dW = None
            dB = None

            for i in range(m):
                if dW is None:
                    dW,dB = self.backward(X[np.newaxis,i,:],y[i])
                else:
                    dtW,dtB = self.backward(X[np.newaxis,i,:],y[i])
                    for j in range(l):
                        dW[j] = dW[j] + dtW[j]
                        dB[j] = dB[j] + dtB[j]

            for i in range(l):
                dW[i] = dW[i]/m
                dB[i] = dB[i]/m
                self.layers[i].W = self.optimizer.update(self.layers[i].W,dW[i])
                self.layers[i].B = self.optimizer.update(self.layers[i].B,dB[i])

            # print(dW)
            # print()



    def predict(self,X):
        m = X.shape[0]
        A = np.zeros(m)

        for i in range(m):
            A[i] = self.forward(X[np.newaxis,i,:])

        return A
