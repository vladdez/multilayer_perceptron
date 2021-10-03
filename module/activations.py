import numpy as np
from module.layers import Layer
import warnings

warnings.filterwarnings('ignore')


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return 1.0 / (1.0 + np.exp(-input))

    def backward(self, input, grad_output):
        p = self.forward(input)
        return grad_output * (p * (1 - p))  # штраф за неувернность


class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return np.maximum(0, input)

    def backward(self, input, grad_output):
        return grad_output * (input > 0).astype(int)


class SoftMax(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        exp = np.exp(input - np.max(input, axis=1)[:, np.newaxis])
        div = np.sum(exp, axis=1)[:, np.newaxis]
        return exp / div

    def backward(self, input, grad_output):
        p = self.forward(input)
        return grad_output * (p * (1 - p))  # штраф за неувернность
