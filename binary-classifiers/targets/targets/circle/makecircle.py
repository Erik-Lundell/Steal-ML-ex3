from sklearn.datasets import make_circles
from sklearn.datasets import dump_svmlight_file
from sklearn.cross_validation import train_test_split

X1, Y1 = make_circles(n_samples=5000, noise=0.07, factor=0.4)

X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=.25)

f_train = open('train', 'w')
dump_svmlight_file(X_train, y_train, f_train, zero_based=False)
f_train.close()

f_test = open('test', 'w')
dump_svmlight_file(X_test, y_test, f_test, zero_based=False)
f_test.close()