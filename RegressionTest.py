import numpy as np
from nn.Model import Model
from nn.Activations import Linear,ReLU
from nn.Layers import Dense
from nn.Loss import MeanSquared
from nn.Optimizers import StochasticGradientDescent
import matplotlib.pyplot as plt

x = np.linspace(-2 * np.pi, 2 * np.pi, 800)
y = np.sin(x) + 0.1 * np.random.randn(len(x))

xt = np.linspace(-2 * np.pi, 2 * np.pi, 200)
yt = np.sin(xt) + 0.1 * np.random.randn(len(xt))

# Plot Train
plt.figure(figsize=(10, 6))
plt.scatter(x, y)
plt.title("Sine Function with Random Noise")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

print(x.shape,y.shape)

X_train = x.reshape(-1,1)
y_train = y.reshape(-1,1)

X_test = xt.reshape(-1,1)
# normalisation
max_val = np.max(X_train)
max_val = max(max_val,np.max(X_test))

X_train = X_train / max_val
X_test = X_test / max_val


model = Model([
    Dense(units=128,in_features=1,activation=ReLU()),
    Dense(units=1,in_features=128,activation=Linear())
])

model.compile(loss=MeanSquared(),optimizer=StochasticGradientDescent(alpha=0.05))

model.fit(X_train,y_train,epoch=10000)
y_pred = model.predict(X_test)
print(y_pred)

yp = y_pred.flatten()

# Plot pred
plt.figure(figsize=(10, 6))
plt.scatter(X_test, yt)
plt.plot(X_test,y_pred,color='red')
plt.title("Sine Function Predicted VS Actual")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
