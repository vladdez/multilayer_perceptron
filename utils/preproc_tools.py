import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


def my_round(array, thrs: float = 0.5, activation: str = 'sigmoid'):
    if activation == 'softmax':
        return np.argmax(array, axis=1)
    return (array.flatten() > thrs).astype(int)


class StandardScaler:
    def __init__(self):
        self.mean = {}
        self.std = {}

    def get_scales(self, data):
        for i in data.columns:
            self.mean[i] = np.mean(data[i].values)
            self.std[i] = np.std(data[i].values)

    def scale(self, value, mean, std):
        return (value - mean) / std

    def transform(self, data):
        data = data.copy()
        for i in data.columns:
            data[i] = data[i].apply(self.scale, mean=self.mean[i], std=self.std[i])
        return data

    def fit_transform(self, data):
        self.get_scales(data)
        return self.transform(data)


class LabelMulticlassEncoder:
    def __init__(self):
        self.mapping = None

    def fit_transform(self, targets):
        classes = np.unique(targets)
        self.mapping = {classes[i]: i for i in range(len(classes))}
        encoded = np.zeros((targets.shape[0], 2), dtype=int)
        for i, t in enumerate(targets):
            encoded[i][self.mapping[t]] = 1
        return encoded


class LabelEncoder:
    def __init__(self):
        self.mapping = None

    def fit_transform(self, targets):
        classes = np.unique(targets)
        self.mapping = {classes[i]: i for i in range(len(classes))}
        encoded = np.zeros(targets.shape, dtype=int)
        for i, j in self.mapping.items():
            encoded[targets == i] = j
        return encoded

def save_model(model):
    with open('saved/model.pkl', 'wb') as file:
        pickle.dump(model, file)


def load_model():
    with open('saved/model.pkl', 'rb') as file:
        return pickle.load(file)


def drop_correlated(data):
    m = data.corr().abs()
    mask = np.triu(np.ones(m.shape), k=1).astype(np.bool)
    m = m.where(mask)
    labels = [i for i in m.columns if any(m[i] > 0.9)]
    return data.drop(columns=labels), labels


def load_csv(name):
    try:
        d = pd.read_csv(name)
    except FileNotFoundError as e:
        print(e)
        exit()
    except pd.errors.EmptyDataError as e:
        print(e)
        exit()
    return d


def plot_curves(train_list, val_list, type):
    plt.plot(np.arange(len(train_list)), train_list, color='blue', label='train')
    plt.plot(np.arange(len(val_list)), val_list, color='red', label='val')
    plt.legend(loc='best')
    plt.title(f'Development of {type} during training')
    plt.xlabel('Number of iterations')
    plt.ylabel(type)
    plt.savefig(f'plots/{type.lower()}.png')
    plt.close()


def check_best_model(metric_list: list, current_metric: float) -> bool:
    if len(metric_list) == 0:
        return True
    if min(metric_list) > current_metric:
        return True
    return False
