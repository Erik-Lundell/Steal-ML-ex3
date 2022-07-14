__author__ = 'Fan'

import os
import copy
import logging
from select import select

import numpy as np

from sklearn import svm
from sklearn.datasets import load_svmlight_file

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from algorithms.OnlineBase import OnlineBase


class LordMeek(OnlineBase):
    def __init__(self, target, test_xy, error=None, delta=None):
        self.X_test, self.y_test = test_xy
        super(self.__class__, self).__init__('LM', +1, -1, target, len(self.X_test[0]), 'uniform', error)

        self.e = error
        self.delta = delta

        if 0 in self.y_test:
            self.NEG = 0
        elif -1 in self.y_test:
            self.NEG = -1
        else:
            print ('Watch out for test file! Neither 0 nor 1 is included!')

    def find_starters(self):
        """
        This function finds a pair of instances. One positive and one negative
        :param clf: classifier being extracted
        :return: (x+, x-) a pair of instances
        """
        # perdict = 1 ? inner(x, coef) + intercept_ > 0 : 0

        x_n, x_p = (None, None)
        x_n_found = False
        x_p_found = False
        for d in self.X_test:
        
            if x_n_found and x_p_found:
                break

            if self.query(d) == 1 and (not x_p_found):
                x_p = d
                x_p_found = True
            elif self.query(d) == self.NEG and (not x_n_found):
                x_n = d
                x_n_found = True

        return x_p, x_n

    def find_witness(self):
        x_p, x_n = self.find_starters()
        assert x_p is not None and self.query(x_p) == 1
        assert x_n is not None and self.query(x_n) == self.NEG
        dim = len(x_p)
        assert dim == len(x_n)

        last_p = -1
        for i in range(0, dim):
            # record the old value
            last_x_p_i = x_p[i]
            # change the value
            x_p[i] = x_n[i]
            if self.query(x_p) == self.NEG:
                # if flips
                last_x_p = copy.copy(x_p)
                last_x_p[i] = last_x_p_i
                assert self.query(x_p) == self.NEG and self.query(last_x_p) == 1
                logger.debug('witness found for dim %d' % i)
                return i, last_x_p, x_p

        return None

    def line_search(self, x, i):
        """
        starting at x (a negative point), search along dimension i, find a point very close to boundary
        :param x: starting point
        :param i: dimension to search
        :return: return the point near boundary
        """
        # make sure to start at a negative point
        assert self.query(x) == self.NEG
        # detach
        new_x = copy.copy(x)

        # phase II: binary search between init and x[i]
        def b(l, r):
            # print 'binary search [%f, %f]' % (l, r)
            # c(l) = 1 && c(r) = 0
            m = 0.5 * (l + r)
            new_x[i] = m
            if self.query(new_x) == self.NEG:
                return b(l, m)
            else:
                if abs(l - m) < self.e:
                    return m, abs(l - m)
                return b(m, r)

        # phase I: exponential explore
        init_xi = x[i]
        step = 1.0 / 100

        # TODO not float64 yet
        while new_x[i] < np.finfo('f').max:
            new_x[i] += step
            if self.query(new_x) == 1:
                return b(new_x[i], init_xi)

            new_x[i] = init_xi
            new_x[i] -= step
            if self.query(new_x) == 1:
                return b(new_x[i], init_xi)

            step *= 2
    
    def find_continous_weights(self):
        f, sp, sn = self.find_witness()
        sp_f = sp[f]
        sn_f = sn[f]
        w_f = 1.0 * np.sign(sp_f - sn_f)
        x0, _ = self.push_to_b(sn, sp, self.e)

        # get a x1 with gap(x0,x1) = 1 & c(x1) = 0
        x1 = copy.copy(x0)
        x1[f] = x1[f] - w_f

        w = np.zeros(len(x0))  # target
        w[f] = w_f
        for i in range(0, len(x0)):
            if i == f:
                continue

            # Saftey-check whether we can switch sign within our given delta. (otherwise, assume w_i = 0)
            u = np.zeros(len(x0)) # unit vector
            u[i] = 1.0
            a = np.add(x1, u / self.delta)
            b = np.add(x1, -u / self.delta)
            if self.query(a) == self.query(b):
                w[i] = 0

            #Otherwise performe line search
            else:
                logger.debug('Line search for dim %d', i)
                new_x_i, err = self.line_search(x1, i)
                w[i] = 1.0 / (new_x_i - x1[i])

        #Return weights and dim f used in initial sign witness.
        return w, f

    def benchmark_old(self, w, f):
        b = self.clf1.intercept_ / self.clf1.coef_[0][f]

        error_clf = 0.0 
        error_lrn = 0.0
        for test_x, test_y in zip(self.X_test, self.y_test):
            test_x = np.reshape(test_x,(1,-1))

            extracted_y = self.POS if np.inner(w, test_x) + b > 0 else self.NEG
            queried_y = self.clf1.predict(test_x)
            if extracted_y != test_y:
                error_lrn += 1

            if queried_y != test_y:
                error_clf += 1

        #R_test = fraction where extracted_y matches tested_y (definition from paper)
        #R_unif = R_test but sample from random input vectors (definition from paper) 
        # Accuracy = 1-R
        pe_clf = 1 - error_clf/ len(self.y_test)
        pe_lrn = 1 - error_lrn/ len(self.y_test)
        
        #No idea what these metrics are...
        print ('L_test = %f' % max(pe_clf - pe_lrn, .0))
        print ('L_unif = %f' % (0.0,))

        return

    def compare_weights(self, w):
        classifier_w = np.reshape(self.clf1.coef_,(1,-1))

        #normalize estimated so that the total length is same as classifier weights
        #For classifying, scaling does not matter
        w_norm = w*np.sqrt(np.inner(classifier_w,classifier_w)) / np.sqrt(np.inner(w,w))

        fraction = np.maximum(classifier_w/w_norm, w_norm/classifier_w)
        fraction = [i[0] for i in fraction]
        print(f"Fraction difference {fraction}")
        print(f"Error bound: {1+self.e}")
        print(f"Passes: {np.array(fraction) < 1+self.e}")

        return np.max(fraction)

    def test(self):
        w, f = self.find_continous_weights()
        fraction = self.compare_weights(w)

        # Original old test uses intercept of classifier in extracted prediction...
        # Orginal test used 
        self.benchmark_old(w,f)

        return self.get_n_query(), fraction


if __name__ == '__main__':

    #Load training data and train black box model
    X_train, y_train = load_svmlight_file('targets/targets/diabetes/test.scale', n_features=8)
    X_train = X_train.todense().tolist()
    clf = svm.LinearSVC()
    clf.fit(X_train, y_train)
    n_features = len(X_train[0])

    # Perform Lowd-Meek
    X_test, y_test = load_svmlight_file('targets/targets/diabetes/test.scale', n_features=8)
    X_test  = X_test.todense().tolist()

    epsilons = (0.5, .1, .01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)

    # Data extracted 
    num_queries = [] #Number of queries
    w_fraction_errors = []
    for e in epsilons:
        delta = 1.0 / 10000

        # e defines the allowed error in the line search, finally giving us |w_est/w| <= 1+epsilon for each weight
        # delta basically defines a bound on how close to 0 weights can be (not super important).
        ex = LordMeek(clf, (X_test, y_test), error=e, delta=delta)

        nq, f = ex.test()
        num_queries.append(nq)
        w_fraction_errors.append(f)

    # This plots the difference between 1+e and the biggest ration between w_i and estimated w_i
    # Gives an upper bound on the ratio error of the weights
    # Plotted as a fraction of 1+e, i.e interpreted as % of error bound
    # If it is larger than 0, the estimation is within the bound, bigger is better
    # 20% = largest weight error is still 20% better than allowed error 
    epsilon_frac = 1+ np.array(epsilons)
    plt.plot(epsilons,(epsilon_frac - w_fraction_errors) / epsilon_frac * 100)
    plt.xscale("log")
    plt.title("Upper bound on weight error")
    plt.xlabel("Epsilon [-]")
    plt.ylabel("Relative error gap [%]")

    plt.show()