import numpy as np
import matplotlib.pyplot as plt
from main.Model import Model
from main.Activations import Sigmoid,ReLU,Softmax,Linear
from main.Layers import Dense
from main.Loss import SparseCategoricalCrossEntropy,BinaryCrossEntropy,MeanSquared
from main.Optimizers import GradientDescent


model = Model([
    Dense(units=128,in_features=1,activation=ReLU()),
    Dense(units=256,in_features=128,activation=ReLU()),
    Dense(units=128,in_features=256,activation=ReLU()),
    Dense(units=1,in_features=128,activation=Linear())
])

model.compile(loss=MeanSquared(),optimizer=GradientDescent(alpha=0.1))

X_train = np.linspace(-4 * np.pi, 4 * np.pi, 1000)
y_train = np.sin(X_train) + 0.1 * np.random.randn(len(X_train))
X_train = X_train[np.newaxis].T
model.fit(X_train,y_train,epoch=1000)

y_pred = model.predict(X_train)

plt.plot(X_train,y_pred,color='red')
plt.scatter(X_train,y_train)
plt.show()
# print(X_train)
# print(y_train,y_pred)



# print(X_train[np.newaxis,0,:])
