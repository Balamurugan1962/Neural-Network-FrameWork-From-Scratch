import numpy as np
from main.Model import Model
from main.Activations import Softmax
from main.Layers import Dense
from main.Loss import CategoricalCrossEntropy
from main.Optimizers import StochasticGradientDescent


X_train = np.array([[0.5, -0.2]])
y_train = np.array([[0, 1, 0]])

model = Model([Dense(units=3,in_features=2,activation=Softmax())])

model.compile(loss=CategoricalCrossEntropy(),optimizer=StochasticGradientDescent(alpha=0.1))

model.fit(X_train,y_train,epoch=100)
y_pred = model.predict(X_train)
print(y_pred)
