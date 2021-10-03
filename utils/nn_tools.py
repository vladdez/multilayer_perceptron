import numpy as np
from sklearn.model_selection import train_test_split


def batcher(x, y, batchsize: int = 32, shuffle: bool = False):
    assert len(x) == len(y)
    if shuffle:
        x_sh = np.random.permutation(len(x))
    for i in range(0, len(x) - batchsize + 1, batchsize):
        if shuffle:
            b = x_sh[i:i + batchsize]
        else:
            b = x[i:i + batchsize]
        yield x[b], y[b]


def split_to_train_val_test(x, y):
    X_train_full, X_test, y_train_full, y_test = train_test_split(x, y, test_size=float(0.33), random_state=int(42))
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=float(0.33),
                                                      random_state=int(42))
    return X_train, y_train, X_val, y_val, X_test, y_test


def split_to_train_val(x, y):
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=float(0.5),
                                                      random_state=int(42))
    return X_train, y_train, X_val, y_val


def init_weights(params, w_method: str = 'xavier_normal', b_method: str = 'ones'):
    for layer in params:
        if w_method == 'zeros':
            layer.weights = np.zeros(layer.w_shape)
        elif w_method == 'normal':
            layer.weights = np.random.normal(loc=0., scale=1., size=layer.w_shape)
        elif w_method == 'xavier_normal':
            layer.weights = np.random.normal(loc=0., scale=(np.sqrt(layer.w_shape[0])), size=layer.w_shape)
        elif w_method == 'kaiming_normal':
            layer.weights = np.random.normal(loc=0., scale=(np.sqrt(2 / layer.w_shape[0])), size=layer.w_shape)
        else:
            raise NotImplementedError

        if b_method == 'zeros':
            layer.biases = np.zeros(layer.b_shape)
        elif b_method == 'ones':
            layer.biases = np.ones(layer.b_shape)
        elif b_method == 'normal':
            layer.biases = np.random.normal(loc=0., scale=1., size=layer.b_shape)
        else:
            raise NotImplementedError


def confusion_matrix(y_true, y_pred, mode: str = 'matrix'):
    y_pred = y_pred.astype(int)
    unique_classes = np.unique(y_true)
    if y_true.shape != y_pred.shape:
        raise Exception(
            f'y_true and y_pred should be equal in shape, but y_true shape is {y_true.shape} and y_pred shape is {y_pred.shape}')
    len_unique = len(unique_classes)
    if len_unique <= 1:
        raise Exception(
            f'in y_true should be more than 1 class, but y_true shape is {y_true.shape} and y_true is {y_true}')
    conf = np.zeros(shape=(len_unique, len_unique))
    for index, item in enumerate(y_true):
        conf[item, y_pred[index]] += 1
    if mode == 'all':
        conf_dict = {}
        for label in range(len_unique):
            tp = conf[label, label]
            fp = np.sum(conf[label, :]) - conf[label, label]
            fn = np.sum(conf[:, label]) - conf[label, label]
            tn = np.sum(conf) - (fp + fn + tp)
            conf_dict[label] = (tp, tn, fp, fn)
        return conf, conf_dict
    return conf


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred):
    conf, conf_dict = confusion_matrix(y_true, y_pred, mode='all')
    tp, tn, fp, fn = conf_dict[0]
    return tp / (tp + fp + 0.0001)


def recall_score(y_true, y_pred):
    conf, conf_dict = confusion_matrix(y_true, y_pred, mode='all')
    tp, tn, fp, fn = conf_dict[0]
    return tp / (tp + fn + 0.0001)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray):
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = (2 * prec * rec) / (prec + rec + 0.0001)
    return f1


class EarlyStopping:
    def __init__(self, esr: int = 5):
        self.esr = esr
        self.current_esr = 0
        self.loss_hist = []

    def add_loss(self, loss: float):
        self.loss_hist.append(loss)

    def check_stop_training(self):
        if self.current_esr > self.esr:
            print(f'Models early stopping rate is higher than {self.esr}. Stop training, model would not be better')
            return True
        if len(self.loss_hist) > 1:
            if min(self.loss_hist[:-1]) < self.loss_hist[-1]:
                self.current_esr += 1
            else:
                self.current_esr = 0
        return False
