import numpy as np


class HeUniform:
    def __call__(self,in_features,units):
        limit = np.sqrt(6/in_features)
        return np.random.uniform(-limit,limit,(in_features, units))


class XavierUniform:
    def __call__(self,in_features,units):
        limit = np.sqrt(6 / (in_features + units))
        return np.random.uniform(-limit, limit, (in_features, units))
