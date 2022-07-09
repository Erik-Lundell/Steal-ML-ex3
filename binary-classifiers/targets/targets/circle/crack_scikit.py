import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn import svm
from sklearn.metrics import accuracy_score
import sys

np.set_printoptions(threshold=np.nan)

sys.path.append('../..')
from algorithms.OnlineBase import OnlineBase
from algorithms.RBFTrainer import RBFKernelRetraining


def main():
    X1, Y1 = make_circles(n_samples=800, noise=0.07, factor=0.4)
    frac0 = len(np.where(Y1 == 0)[0]) / float(len(Y1))
    frac1 = len(np.where(Y1 == 1)[0]) / float(len(Y1))

    print "Percentage of '0' labels:", frac0
    print "Percentage of '1' labels:", frac1

    plt.figure()
    plt.subplot(121)
    plt.title(
        "Our Dataset: N=200, '0': {0} '1': {1} ".format(
            frac0,
            frac1),
        fontsize="large")

    plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)
    plt.xlim((-2, 2))
    plt.ylim((-2, 2))

    clf = svm.SVC()
    clf.fit(X1, Y1)

    print accuracy_score(Y1, clf.predict(X1))

    ex = OnlineBase('circle', 1, 0, clf.predict, 2, 'uniform', .1)
    step = 6
    train_x, train_y = [], []
    val_x, val_y = [], []
    while True:
        ex.collect_pts(step)
        train_x.extend(ex.pts_near_b)
        train_y.extend(ex.pts_near_b_labels)
        val_x.extend(ex.support_pts)
        val_y.extend(ex.support_labels)
        try:
            e = RBFKernelRetraining('circle', train_x, train_y, val_x, val_y, train_x, train_y, n_features=2)
            print ex.get_n_query(), e.grid_retrain_in_x()
        except KeyboardInterrupt:
            print 'Done'
            break

    train_x = np.array(train_x)
    plt.subplot(122)
    plt.scatter(train_x[:, 0], train_x[:, 1], c=train_y)
    plt.xlim((-2, 2))
    plt.ylim((-2, 2))
    plt.show()


if __name__ == '__main__':
    main()
