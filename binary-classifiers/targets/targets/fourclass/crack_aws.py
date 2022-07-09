from algorithms.awsOnline import AWSOnline
from algorithms.RBFTrainer import RBFKernelRetraining
from sklearn.datasets import load_svmlight_file

val_name = ['x1', 'x2']
n_features = 2

ex = AWSOnline(model_id='ml-Je6DdX8c57P', label_p=1, label_n=0,
               n_features=n_features, val_name=val_name, ftype='uniform', error=.1)

step = 6
train_x, train_y = [], []
val_x, val_y = [], []
test_x, test_y = load_svmlight_file('test.scale', n_features)
test_x = test_x.todense()
test_y = [a if a == 1 else 0 for a in test_y]
try:
    while True:
        ex.collect_pts(step)
        train_x.extend(ex.pts_near_b)
        train_y.extend(ex.pts_near_b_labels)
        val_x.extend(ex.support_pts)
        val_y.extend(ex.support_labels)
        e = RBFKernelRetraining('', train_x, train_y, val_x, val_y, test_x, test_y, n_features)
        print ex.get_n_query(), e.grid_retrain_in_x()
except KeyboardInterrupt:
    print 'Done'
