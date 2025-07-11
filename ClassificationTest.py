import numpy as np

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from nn.Model import Model
from nn.Activations import Sigmoid,ReLU,Softmax,Linear
from nn.Layers import Dense
from nn.Loss import CategoricalCrossEntropy,BinaryCrossEntropy,MeanSquared
from nn.Optimizers import StochasticGradientDescent

centers = [(-10, -10), (10, 10)]
cluster_std = 5
random_state_train = 42
random_state_test = 99

X_train, y_train = make_blobs(n_samples=8000, centers=centers,cluster_std=cluster_std, random_state=random_state_train)

X_test, y_test = make_blobs(n_samples=2000, centers=centers,cluster_std=cluster_std, random_state=random_state_test)

fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax2.scatter(X_test[y_test==0,0],X_test[y_test==0,1])
ax2.scatter(X_test[y_test==1,0],X_test[y_test==1,1])
ax2.set_title('Test Values')
ax1.scatter(X_train[y_train==0,0],X_train[y_train==0,1])
ax1.scatter(X_train[y_train==1,0],X_train[y_train==1,1])
ax1.set_title('Train Values')
plt.show()



# Since y is of (m,) shape we make it as Tensor
y_train_tensor = y_train.reshape(-1, 1)
y_test_tensor = y_test.reshape(-1, 1)

model = Model([
    Dense(units=10,in_features=2,activation=ReLU()),
    Dense(units=1,in_features=10,activation=Sigmoid())
])

model.compile(loss=BinaryCrossEntropy(),optimizer=StochasticGradientDescent(alpha=0.05))

model.fit(X_train,y_train_tensor,epoch=100)

y_pred_tensor = model.predict(X_test)

y_pred = y_pred_tensor.flatten()
y_pred[y_pred>=0.5] = 1
y_pred[y_pred<0.5] = 0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax2.scatter(X_test[y_test==0,0],X_test[y_test==0,1])
ax2.scatter(X_test[y_test==1,0],X_test[y_test==1,1])
ax2.set_title('Test Values')
ax1.scatter(X_test[y_pred==0,0],X_test[y_pred==0,1])
ax1.scatter(X_test[y_pred==1,0],X_test[y_pred==1,1])
ax1.set_title('Predicted Values')
plt.show()


correct = np.sum(y_test == y_pred)
accuracy = (correct / len(y_test)*100)
print(f"Test Accuracy: {accuracy:.2f}%")
