import numpy as np
import copy


class Optimizer:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def action(self, iter_num):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, lr)
        self.params = params
        self.lr = lr

    def action(self, iter_num):
        for layer in self.params:
            layer.weights -= self.lr * layer.w_grad
            layer.biases -= self.lr * layer.b_grad


class Momentum(Optimizer):
    def __init__(self, params, learning_rate):
        super().__init__(params, learning_rate)
        self.model_params = params
        self.lr = learning_rate
        self.momentum = 0.99
        self.w_velocities = {i: np.zeros_like(self.model_params[i].weights) for i in
                             range(len(self.model_params))}
        self.b_velocities = {i: np.zeros_like(self.model_params[i].biases) for i in
                             range(len(self.model_params))}

    def action(self, iter_num):
        for index, layer in enumerate(self.model_params):
            self.w_velocities[index] = self.momentum * self.w_velocities[index] + \
                                       self.lr * layer.w_grad
            self.b_velocities[index] = self.momentum * self.b_velocities[index] + \
                                       self.lr * layer.b_grad
            layer.weights -= self.w_velocities[index]
            layer.biases -= self.b_velocities[index]

"""
Adaptive momemtum optimizer
https://arxiv.org/pdf/1412.6980.pdf
https://habr.com/ru/post/318970/
"""

class Adam(Optimizer):
    def __init__(self, params, learning_rate, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, learning_rate)
        self.model_params = params
        self.lr = learning_rate
        self.betas = betas
        self.eps = eps

        params_len = len(self.model_params)

        self.w_m = {i: np.zeros_like(self.model_params[i].weights) for i in range(params_len)}
        self.b_m = {i: np.zeros_like(self.model_params[i].biases) for i in range(params_len)}
        self.w_v = copy.deepcopy(self.w_m)
        self.b_v = copy.deepcopy(self.b_m)

    def action(self, iter_num):
        for index, layer in enumerate(self.model_params):
            self.w_m[index] = self.betas[0] * self.w_m[index] + (1. - self.betas[0]) * layer.w_grad
            self.b_m[index] = self.betas[0] * self.b_m[index] + (1. - self.betas[0]) * layer.b_grad

            self.w_v[index] = self.betas[1] * self.w_v[index] + (1. - self.betas[1]) * layer.w_grad ** 2
            self.b_v[index] = self.betas[1] * self.b_v[index] + (1. - self.betas[1]) * layer.b_grad ** 2

            w_m_hat = self.w_m[index] / (1. - self.betas[0] ** iter_num)
            b_m_hat = self.b_m[index] / (1. - self.betas[0] ** iter_num)

            w_v_hat = self.w_v[index] / (1. - self.betas[1] ** iter_num)
            b_v_hat = self.b_v[index] / (1. - self.betas[1] ** iter_num)

            layer.weights -= self.lr * w_m_hat / (np.sqrt(w_v_hat) + self.eps)
            layer.biases -= self.lr * b_m_hat / (np.sqrt(b_v_hat) + self.eps)
