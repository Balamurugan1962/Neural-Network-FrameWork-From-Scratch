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


# For now i just made for whole Train Set,
# and everytime it backtracks for single testcase(Half works with Batch but need to generalise for eveything)
# In feature need to Implement Batch , Mini Batch Training
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

# Need to Clean code this part
    def fit(self,X,y,epoch=100):
        m = X.shape[0]
        l = len(self.layers)
        for e in range(epoch):
            if (e+1) % 10 == 0:
                y_hat = self.predict(X)
                avg_loss = np.mean(self.loss(y_hat, y))
                print(f"Epoch {e + 1:>5}: Loss = {avg_loss:.6f}")

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
                self.layers[i].W = self.optimizer.update(self.layers[i].W,dW[i],i*100)
                self.layers[i].B = self.optimizer.update(self.layers[i].B,dB[i],i*100+1)

            # print(dW)
            # print()



    def predict(self,X):
        m = X.shape[0]
        A = []

        for i in range(m):
            A.append(self.forward(X[np.newaxis,i,:]))

        return np.vstack(A)
