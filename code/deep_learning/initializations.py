import numpy as np
import math

def constant_value(value, shape):
	if shape is None:
		shape = ()
	res = np.ones(shape) * value
	return res

def random_normal(scale, shape, mean):
	res = scale*np.random.randn(shape) + mean
	return res

def random_uniform(low, high, shape):
	res = (high - low) * np.random.random(shape) + low
	return res

def xavier_normal(shape):
	n_in = shape[0]
	n_out = shape[1]
	limit = math.sqrt(2/(n_in + n_out))
	return np.random.randn(shape)*limit

def xavier_uniform(shape):
	n_in = shape[0]
	n_out = shape[1]
	limit = math.sqrt(6/(n_in + n_out))
	# limit = math.sqrt(1/n_in)
	return np.random.uniform(-limit, limit, shape)


