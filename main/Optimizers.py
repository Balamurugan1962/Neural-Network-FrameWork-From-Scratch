import numpy as np


class GradientDescent:
    def __init__(self,alpha=0.001):
        self.alpha = alpha

    def update(self,x,x_grad,key):
        return x - self.alpha * x_grad
