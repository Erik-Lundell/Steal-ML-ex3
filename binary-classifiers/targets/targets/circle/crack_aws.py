import matplotlib.pyplot as plt
from algorithms.awsOnline import AWSOnline
import numpy as np
from algorithms.RBFTrainer import RBFKernelRetraining

for i in range(0, 1):
    val_name = ['x1', 'x2']

    ex = AWSOnline('ml-i0GeYZaGQ3f', 1, 0, 2, val_name, 'uniform', .1)

    step = 6
    train_x, train_y = [], []
    val_x, val_y = [], []
    try:
        while True:
            ex.collect_pts(step)
            train_x.extend(ex.pts_near_b)
            train_y.extend(ex.pts_near_b_labels)
            val_x.extend(ex.support_pts)
            val_y.extend(ex.support_labels)
            e = RBFKernelRetraining('circle', train_x, train_y, val_x, val_y, train_x, train_y, n_features=2)
            print ex.get_n_query(), e.grid_retrain_in_x()
    except KeyboardInterrupt:
        print 'Done'

    train_x = np.array(train_x)
    plt.figure()
    plt.scatter(train_x[:, 0], train_x[:, 1], c=train_y)
    plt.xlim((-2, 2))
    plt.ylim((-2, 2))
    plt.show()
