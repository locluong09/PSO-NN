import numpy as np
import math

class SquareLoss():

	def loss(self, y, y_predict):
		length = len(y)
		re = 1/(2*length)*(np.power(y - y_predict, 2))
		# print(re.shape)
		return re

	def derivative(self, y, y_predict):
		length = len(y)
		return 1/(length)*(y_predict - y)

	def RMSE(self, y, y_predict):
		return math.sqrt(np.mean(np.power(y - y_predict, 2)))


def accuracy_score(y_true, y_pred):
	accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
	return accuracy

class CrossEntropy():

	def loss(self, y, p):
		p = np.clip(p, 1e-15, 1 - 1e-15)
		return - y * np.log(p) - (1 - y) * np.log(1 - p)

	def derivative(self, y, p):
		p = np.clip(p, 1e-15, 1 - 1e-15)
		return - (y / p) + (1 - y) / (1 - p)

	def RMSE(self, y, p):
		return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))
