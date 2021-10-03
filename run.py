from train import main_train
from test import main_test
from evaluation import main_ev
import os

if __name__ == '__main__':
    for i in range(1, 4):
        print('-------------')
        print('TRY', i)


        os.system("python3.9 evaluation.py")
        os.system("python3.9 train.py  --dataset data_training.csv --activation sigmoid")
        os.system("python3.9 test.py  --dataset data_test.csv")
