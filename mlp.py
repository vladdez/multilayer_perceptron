import numpy as np
import pandas as pd
import matplotlib as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='../datasets/datasets.csv')
    #parser.add_argument('--loss', action="store_true")
    args = parser.parse_args()
    # print(args.__dict__)
    return args.__dict__
    # l_rate = args.__dict__['learning_rate']
    # epochs = args.__dict__['epochs']


def main():
    args = parse_args()
    print(args['dataset'])
    print(argparse)
    print(pd.__version__)
    data = pd.read_cav(args['dataset'])
    #print(datasets.head())
    #LR = LogRe()
    #LR.train(l_rate, epochs, loss)


if __name__ == '__main__':
    main()