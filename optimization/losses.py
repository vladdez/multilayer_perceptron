import numpy as np


class Losser:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(self, y_true, y_pred):
        raise NotImplementedError

    def backward(self, model, y_true, y_pred):
        raise NotImplementedError


class BinaryCrossEntropy(Losser):
    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred):
        target = y_true
        output = np.clip(y_pred, self.eps, 1.0 - self.eps)
        return -1 * np.mean(target * np.log(output) + (1 - target) * np.log(1 - output))

    def backward(self, model, y_true, y_pred):
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        loss_grad = y_pred - y_true[:, np.newaxis]  # Cross_entropy derivative
        model.backward(model.forward_list, loss_grad)


class MeanSquaredErrorLoss(Losser):
    def __call__(self, y_true, y_pred):
        return np.mean(np.square(y_pred - y_true[:, np.newaxis]))

    def backward(self, model, y_true, y_pred):
        loss_grad = -2 * (y_true[:, np.newaxis] - y_pred)
        model.backward(model.forward_list, loss_grad)


class CrossEntropyLoss(Losser):
    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred):
        output = np.clip(y_pred, self.eps, 1. - self.eps)
        output /= output.sum(axis=1)[:, np.newaxis]
        return np.mean(-(y_true * np.log(output)).sum(axis=1))

    def backward(self, model, y_true, y_pred):
        loss_grad = y_pred - y_true
        model.backward(model.forward_list, loss_grad)
