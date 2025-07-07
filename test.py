from tensorflow.keras.datasets import mnist
import numpy as np

from main.Model import Model
from main.Activations import Sigmoid,ReLU,Softmax
from main.Layers import Dense
from main.Loss import SparseCategoricalCrossEntropy,BinaryCrossEntropy
from main.Optimizers import GradientDescent

(X_train, y_train), (X_test, y_test) = mnist.load_data()
np.random.seed(0)
y_test[y_test!=0] = 10
y_test[y_test==0] = 1
y_test[y_test==10] = 0
y_train[y_train!=0] = 10
y_train[y_train==0] = 1
y_train[y_train==10] = 0

x_train = X_train.reshape(X_train.shape[0],-1)
x_test= X_test.reshape(X_test.shape[0],-1)

y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

model = Model([
    Dense(units=128,in_features=784,activation=ReLU()),
    Dense(units=1,in_features=128,activation=Sigmoid())
])

model.compile(loss=BinaryCrossEntropy(),optimizer=GradientDescent(alpha=0.01))
model.fit(x_train,y_train,epoch=100)
