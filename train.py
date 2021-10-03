import numpy as np
import pandas as pd
import argparse
import pickle
import random
import os
from utils.preproc_tools import StandardScaler, LabelEncoder, LabelMulticlassEncoder, save_model, load_csv, \
    check_best_model, drop_correlated, plot_curves, my_round
from utils.nn_tools import split_to_train_val_test, split_to_train_val, init_weights, batcher, nn_f1_score, \
    nn_accuracy_score, EarlyStopping
from module.activations import Sigmoid, ReLU, SoftMax
from module.model import Model
from module.layers import Dense
from optimization.optimizers import SGD, Momentum, Adam
from optimization.losses import BinaryCrossEntropy, CrossEntropyLoss


def seed_everything():
    random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)
    np.random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='datasets/data.csv')
    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--epochs', default=200)
    parser.add_argument('--batchsize', default=32)
    parser.add_argument('--perf', action="store_true")
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--activation', default='sigmoid')
    args = parser.parse_args()
    return args.__dict__


def main_train():
    args = parse_args()
    seed_everything()
    scaler = StandardScaler()
    data = load_csv(args['dataset'])
    activation = args['activation']
    encoder = LabelEncoder()
    if activation == 'softmax':
        encoder = LabelMulticlassEncoder()
    data = pd.DataFrame(data.to_numpy(), columns=[str(i + 1) for i in range(data.shape[1])])
    y = encoder.fit_transform(data['2'])
    x = data.drop(columns=['1', '2'], axis=1)
    x = x.astype(np.float64)
    x = scaler.fit_transform(x)
    x, dropped = drop_correlated(x)
    x = x.to_numpy()
    if args['dataset'] == 'datasets/data.csv':
        X_train, y_train, X_val, y_val, X_test, y_test = split_to_train_val_test(x, y)
        with open('saved/test_data.pkl', 'wb') as f:
            pickle.dump((X_test, y_test), f, protocol=4)
    else:
        X_train, y_train, X_val, y_val = split_to_train_val(x, y)

    hidden_size = 60
    output_size = 1
    last_activation = Sigmoid()
    if activation == 'softmax':
        output_size = 2
        last_activation = SoftMax()
    model = Model(activation=activation, dropped=dropped)
    model.add_layer(Dense(X_train.shape[1], hidden_size * 5))
    model.add_layer(ReLU())
    model.add_layer(Dense(hidden_size * 5, hidden_size * 3))
    model.add_layer(ReLU())
    model.add_layer(Dense(hidden_size * 3, hidden_size))
    model.add_layer(ReLU())
    model.add_layer(Dense(hidden_size, int(hidden_size / 2)))
    model.add_layer(ReLU())
    model.add_layer(Dense(int(hidden_size / 2), output_size))
    model.add_layer(last_activation)
    es = EarlyStopping(esr=15)
    init_weights(model.params_with_grad, 'kaiming_normal')

    losser = BinaryCrossEntropy()
    if activation == 'softmax':
        losser = CrossEntropyLoss()
    optimizer = Adam(model.params_with_grad, float(args['lr']))

    save_model(model)
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for i in range(int(args['epochs'])):
        train_loss_b = []
        val_loss_b = []
        train_acc_b = []
        val_acc_b = []
        for x_batch_train, y_batch_train in batcher(X_train, y_train, batchsize=int(args['batchsize']), shuffle=True):
            preds = model.forward(x_batch_train)
            loss = losser(y_batch_train, preds)
            accuracy = nn_f1_score(my_round(y_batch_train, activation=model.activation),
                                   my_round(preds, activation=model.activation))
            train_loss_b.append(loss)
            train_acc_b.append(accuracy)
            losser.backward(model, y_batch_train, preds)
            optimizer.action(i + 1)
            model.clear_cache()

        for x_batch_val, y_batch_val in batcher(X_val, y_val, batchsize=int(args['batchsize']), shuffle=True):
            preds = model.forward(x_batch_val)
            loss = losser(y_batch_val, preds)
            accuracy = nn_f1_score(my_round(y_batch_val, activation=model.activation),
                                   my_round(preds, activation=model.activation))
            val_loss_b.append(loss)
            val_acc_b.append(accuracy)
            model.clear_cache()

        if args['perf']:
            print(f'epoch {i + 1} / {int(args["epochs"])}, loss: {round(np.mean(train_loss_b), 2)}, '
                  f'val_loss: {round(np.mean(val_loss_b), 2)}, f1: {round(np.mean(train_acc_b), 2)}, '
                  f'val_f1: {round(np.mean(val_acc_b), 2)}')

        if check_best_model(val_acc, np.mean(val_acc_b)):
            model.save_w()
            save_model(model)
        train_loss.append(np.mean(train_loss_b))
        train_acc.append(np.mean(train_acc_b))
        val_loss.append(np.mean(val_loss_b))
        val_acc.append(np.mean(val_acc_b))

        es.add_loss(np.mean(val_loss_b))
        if es.check_stop_training():
            print(f'epoch {i + 1} / {int(args["epochs"])}, loss: {round(np.mean(train_loss_b), 2)}, '
                  f'val_loss: {round(np.mean(val_loss_b), 2)}, f1: {round(np.mean(train_acc_b), 2)}, '
                  f'val_f1: {round(np.mean(val_acc_b), 2)}')
            break
    if args['plot']:
        plot_curves(train_loss, val_loss, "Loss")
        plot_curves(train_acc, val_acc, "F1")



if __name__ == '__main__':
    main_train()
