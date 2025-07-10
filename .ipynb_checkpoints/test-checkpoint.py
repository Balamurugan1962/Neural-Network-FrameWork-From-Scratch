import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from main.Model import Model
from main.Activations import Sigmoid,ReLU,Softmax,Linear
from main.Layers import Dense
from main.Loss import CategoricalCrossEntropy,BinaryCrossEntropy,MeanSquared
from main.Optimizers import StochasticGradientDescent

centers = [(-5, -5), (5, 5)]
cluster_std = 3.0
random_state_train = 42
random_state_test = 99

X_train, y_train = make_blobs(n_samples=800, centers=centers,cluster_std=cluster_std, random_state=random_state_train)

X_test, y_test = make_blobs(n_samples=200, centers=centers,cluster_std=cluster_std, random_state=random_state_test)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# plt.scatter(X_train[y_train==0,0],X_train[y_train==0,1])
# plt.scatter(X_train[y_train==1,0],X_train[y_train==1,1])
# plt.title("Train Set")
# plt.show()


# print(X_train.shape,y_train.shape)



model = Model([
    Dense(units=10,in_features=2,activation=ReLU()),
    Dense(units=1,in_features=10,activation=Sigmoid())
])

model.compile(loss=BinaryCrossEntropy(),optimizer=StochasticGradientDescent(alpha=0.05))

model.fit(X_train,y_train,epoch=100)

y_pred = model.predict(X_test)

y_pred[y_pred>=0.5] = 1
y_pred[y_pred<0.5] = 0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax2.scatter(X_test[y_test[:,0]==0,0],X_test[y_test[:,0]==0,1])
ax2.scatter(X_test[y_test[:,0]==1,0],X_test[y_test[:,0]==1,1])
ax1.scatter(X_test[y_pred[:,0]==0,0],X_test[y_pred[:,0]==0,1])
ax1.scatter(X_test[y_pred[:,0]==1,0],X_test[y_pred[:,0]==1,1])
plt.show()


correct = np.sum(y_test == y_pred)
accuracy = (correct / len(y_test))
print(f"Test Accuracy: {accuracy:.2f}%")
