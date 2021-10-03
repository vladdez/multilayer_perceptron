import numpy as np


class Layer:
    def __init__(self):
        self.weights = None
        self.biases = None
        self.w_shape = None
        self.b_shape = None
        self.w_grad = None
        self.b_grad = None
        self.require_grad = False

    def forward(self, input):
        raise NotImplementedError

    def backward(self, input, grad_output):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, input_units, output_units):
        super().__init__()
        self.w_shape = (input_units, output_units)
        self.b_shape = (output_units,)
        self.require_grad = True

    def forward(self, input):
        return np.dot(input, self.weights) + self.biases

    def backward(self, input, grad_loss):  # grad_loss = dLoss/dy how loss changes depending on ur prediction
        grad_input = np.dot(grad_loss, self.weights.T)
        self.w_grad = np.dot(input.T, grad_loss)
        self.b_grad = grad_loss.mean(axis=0)

        assert self.w_grad.shape == self.weights.shape and \
               self.b_grad.shape == self.biases.shape
        return grad_input


class Dropout(Layer):
    """
    "Dropout:  A Simple Way to Prevent Neural Networks from Overfitting"
    https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, input: np.ndarray, mode: str = 'train') -> np.ndarray:
        if mode == 'train':
            self.mask = np.random.binomial(1, self.p, input.shape) / self.p
            return input * self.mask
        else:
            return input

    def backward(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * self.mask
