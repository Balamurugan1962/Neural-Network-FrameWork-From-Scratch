# Neural-Networks-From-Scratch
Neural-Network-from-scratch is a not just a plain hardcoded neural architecture, it is a general package like TensorFlow and pyTorch where you can create you custom architecture

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

####
