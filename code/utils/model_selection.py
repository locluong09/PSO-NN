from __future__ import division
import numpy as np
import math as m


def shuffle_(X, y, seed = None):
	''' random samples of dataset X and y'''
	if seed:
		np.random.seed(seed)
	index = np.arange(X.shape[0])
	np.random.shuffle(index)
	return X[index], y[index]
		

def train_test_split(X, y, test_size = 0.2, shuffle = True, seed = None):
	'''split data into train, test set'''
	if shuffle:
		X,y = shuffle_(X, y, seed)
	split = len(y) - int(len(y) // (1/test_size))
	X_train, X_test = X[:split], X[split:]
	y_train, y_test = y[:split], y[split:]

	return X_train,X_test,y_train,y_test

def batch_iterator(X, y=None, batch_size=64):
    """ Simple batch generator """
    n_samples = X.shape[0]
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i+batch_size, n_samples)
        if y is not None:
            yield X[begin:end], y[begin:end]
        else:
            yield X[begin:end]
