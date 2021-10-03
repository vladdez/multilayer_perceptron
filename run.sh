#python -m pip install -r requirements.txt
python .\train.py --plot --perf --dataset data_training.csv
python .\test.py data_test.csv
