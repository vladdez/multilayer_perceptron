import gc
import pickle


class Model:
    def __init__(self, activation, dropped):
        self.network = []
        self.forward_list = []
        self.activation = activation
        self.dropped = dropped

    def __len__(self):
        return len(self.network)

    @property
    def params_with_grad(self):
        return [i for i in self.network if i.require_grad is True]

    def add_layer(self, layer):
        self.network.append(layer)

    def forward(self, input):
        self.forward_list.append(input)
        for layer in self.network:
            self.forward_list.append(layer.forward(self.forward_list[-1]))
        assert len(self.forward_list) == len(self.network) + 1
        pred = self.forward_list[-1]
        return pred

    def backward(self, layer_inputs, loss_grad):
        for layer_i in range(len(self.network))[::-1]:
            layer = self.network[layer_i]
            loss_grad = layer.backward(layer_inputs[layer_i], loss_grad)

    def save_w(self):
        w = []
        # print("Weigths were saved in saved/weights.pkl")
        with open('saved/weights.pkl', 'wb') as file:
            for layer in self.params_with_grad:
                w.append((layer.weights, layer.biases))
            pickle.dump(w, file)

    def load_w(self):
        with open('saved/weights.pkl', 'rb') as file:
            w = pickle.load(file)
        for i, layer in enumerate(self.params_with_grad):
            layer.weights = w[i][0]
            layer.biases = w[i][1]

    def clear_cache(self, collect_garbage: bool = False):
        self.forward_list.clear()
        if collect_garbage:
            for layer in self.params_with_grad:
                layer.weights_grad = None
                layer.biases_grad = None
            gc.collect()
