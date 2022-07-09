import sys
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def prepare_data():
    target = "income"
    data = pd.read_csv('../../../data/adult.csv')

    X = data[list(set(data.columns) - set([target]))]
    y = data[target]

    X = pd.get_dummies(X)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = pd.DataFrame(scaler.fit_transform(X))
    y = pd.Series(LabelEncoder().fit_transform(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = prepare_data()

from sklearn.datasets import dump_svmlight_file

f_train = open('train', 'w')
dump_svmlight_file(X_train, y_train, f_train, zero_based=False)
f_train.close()

f_test = open('test', 'w')
dump_svmlight_file(X_train, y_train, f_test, zero_based=False)
f_test.close()
