# Neural-Networks-From-Scratch
Neural Network from scratch is a not just a plain hardcoded neural architecture, it is a general package like TensorFlow and pyTorch where you can create you custom architecture

This package was mainly inspired from the Tensorflow package, so it follows the same flow like we do in Tensorflow

### Creation of Layer:
For now there is  1 type of layer
#### Dense:

```python3
layer = Dense(units,in_feature,activation)
```
units - number of neurons
in_features - number of features
activation - activation function which is imported from nn.Activations module

### Activation:
currently there are 4 Activation functions

#### ReLU:
 ```python3
 from nn.Activations import ReLU
 layer = Dense(units,in_feature,activation=ReLU())
```

#### Sigmoid:
 ```python3
 from nn.Activations import Sigmoid
 layer = Dense(units,in_feature,activation=Sigmoid())
```

#### Linear:
 ```python3
 from nn.Activations import Linear
 layer = Dense(units,in_feature,activation=Linear())
```

#### Softmax:
 ```python3
 from nn.Activations import Softmax
 layer = Dense(units,in_feature,activation=Softmax())
```

**Note**: Activations are implemented a class so it should be assigned as a object of the class


### Creation of model:
Creation of model is same as we do in Tensorflow, we need to specify the list of layers in sequential is order
```python3
from nn.Model import Model
model = Model(layers)
```
layers : List of Layers

eg :
```python3
from nn.Model import Model
from nn.Activations import Linear

model = Model([
	Dense(units=1,in_features=1,activation=Linear())
])
```

### Compile the Model:
after we initialize the model we need to compile which means we need to specify the Loss and optimizer for the model

```python3
model.compile(loss,optimizer)
```

### Loss
Same like activation, Loss is implemented as Class for abstraction so we need to pass object of class

we have 3 types of Loss:
#### MeanSquared:
```python3
from nn.Loss import MeanSquared
from nn.Model import Model
from nn.Activations import Linear

model = Model([
	Dense(units=1,in_features=1,activation=Linear())
])

model.compile(loss = MeanSquared(),optimizer)

```
#### BinaryCrossEntropy:
```python3
from nn.Loss import BinaryCrossEntropy
from nn.Model import Model
from nn.Activations import Linear

model = Model([
	Dense(units=1,in_features=1,activation=Linear())
])

model.compile(loss = BinaryCrossEntropy,optimizer)
```
#### CategoricalCrossEntropy:
```python3
from nn.Loss import CategoricalCrossEntropy
from nn.Model import Model
from nn.Activations import Linear

model = Model([
	Dense(units=1,in_features=1,activation=Linear())
])

model.compile(loss = CategoricalCrossEntropy,optimizer)
```

**Note:** For now back propagation for CategoricalCrossEntropy loss  is combined with softmax, so for Softmax Activation always use CategoricalCrossEntropy

### Optimizer:
For now we have only one optimisation algorithm implemented, in future it would be expanded and updated

#### StochasticGradientDescent:
```python3
StochasticGradientDescent(alpha)
```
alpha - learning rate (Default it is 0.001)

eg:
```python3
from nn.Loss import CategoricalCrossEntropy
from nn.Model import Model
from nn.Activations import Linear
from nn.Optimizers import StochasticGradientDescent

model = Model([
	Dense(units=1,in_features=1,activation=Linear())
])

model.compile(loss = CategoricalCrossEntropy,optimizer=StochasticGradientDescent())
```

####  Fit:
After compiling now we are ready to train the model, we use **model.fit()** to fit the mode

```python3
model.fit(X_train,y_train,epoch)
```
epoch : number of iterations (Default it is 100)

**Note:**
X_train and y_train should be of nparray and like a Tensor or  dimensions > 1

eg:
```
if X is [1,2,3,4]
it should be [[1],[2],[3],[4]]

if y is [0,0,1]
it should be [[0],[0],[1]]

if X is [[1,2],[1,3]]
it can be same since its dimenstion is more then 1D

same goes to y
```

to convert it with ease use bellow code
```python3
a = a.reshape(-1,1) #Converts 1D to 2D or (m,) to (m,1) shape
```
