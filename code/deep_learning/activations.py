import numpy as np
import math

class Sigmoid():

	def activate(self, x):
		return 1/(1 + np.exp(-x))

	def derivative(self, x):
		return self.activate(x) * (1 - self.activate(x))


class Linear():

	def activate(self, x):
		return x

	def derivative(self, x):
		return 1


class Tanh():

	def activate(self, x):
		return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

	def derivative(self, x):
		return 1 - np.power(self.activate(x),2)


class Relu():

	def activate(self, x):
		return np.where(x >= 0, x, 0)
		# return (x + abs(x))/2

	def derivative(self, x):
		return np.where(x >= 0, 1, 0)
		# return 1. * (x > 0)


class LeakyRelu():
	def __init__(self, alpha = 0.01):
		self.alpha = alpha

	def activate(self, x):
		res = np.where(x >= 0, x, self.alpha * x)
		return res

	def derivative(self, x):
		res = [1 if i >= 0 else self.alpha for i in x]
		return np.array(x)


class Softmax():

	def activate(self, x):
		e_x = np.exp(x - np.max(x, axis = -1, keepdims = True))
		return e_x / np.sum(e_x, axis = -1, keepdims = True)

	def derivative(self, x):
		return self.activate(x) * (1 - self.activate(x))

