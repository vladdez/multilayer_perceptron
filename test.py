import numpy as np
import pandas as pd
import pickle
import argparse
from utils.preproc_tools import StandardScaler, LabelEncoder, LabelMulticlassEncoder, load_model, load_csv, \
    my_round
from utils.nn_tools import f1_score, accuracy_score, precision_score, recall_score
from optimization.losses import BinaryCrossEntropy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='saved/test_data.pkl')
    args = parser.parse_args()
    return args.__dict__


def main_test():
    args = parse_args()
    model = load_model()

    losser = BinaryCrossEntropy()
    encoder = LabelEncoder()
    if model.activation == 'softmax':
        encoder = LabelMulticlassEncoder()
    scaler = StandardScaler()

    if args['dataset'] == 'saved/test_data.pkl':
        with open('saved/test_data.pkl', 'rb') as f:
            X_test, y_test = pickle.load(f)

    else:
        data = load_csv(args['dataset'])
        data = pd.DataFrame(data.to_numpy(), columns=[str(i + 1) for i in range(data.shape[1])])
        y_test = encoder.fit_transform(data['2'])
        x = data.drop(columns=['1', '2'], axis=1)
        x = x.astype(np.float64)
        x = scaler.fit_transform(x)
        X_test = x.drop(columns=model.dropped).to_numpy()

    pred_test = model.forward(X_test)

    print('Loss:', round(losser(y_test, pred_test), 2))
    print('Accuracy:', round(accuracy_score(y_test, my_round(pred_test)), 2))
    print('F1:', round(f1_score(my_round(y_test, activation=model.activation), \
                                   my_round(pred_test, activation=model.activation)), 2))
    print('Presicion:', round(precision_score(my_round(y_test, activation=model.activation),
                                                 my_round(pred_test, activation=model.activation)), 4))
    print('Recall:', round(recall_score(my_round(y_test, activation=model.activation),
                                           my_round(pred_test, activation=model.activation)), 4))

if __name__ == '__main__':
    main_test()
