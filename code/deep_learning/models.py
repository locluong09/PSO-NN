from __future__ import print_function, division
import numpy as np

from utils.model_selection import batch_iterator

class Neural_Networks():
	def __init__(self, optimizer, loss, validation_data = None):
		self.optimizer = optimizer
		self.layers = []
		self.errors = {"training": [], "validation":[]}
		self.loss_function = loss()
		self.validation_set = None
		self.iterations = 0
		if validation_data:
			X, y = validation_data
			self.validation_set = {"X": X, "y": y}

	def add(self, layer):
		'''This method will add a layer to current neural network architecture'''
		if self.layers:
			layer.set_input_shape(shape = self.layers[-1].output_shape())

		if hasattr(layer, 'initialize'):
			layer.initialize()

		if hasattr(layer, 'set_optimizer'):
			layer.set_optimizer(self.optimizer)
			layer.iterations = self.iterations

		self.layers.append(layer)

	def _forward(self, X):
		output = X
		for layer in self.layers:
			output = layer.forward(output)
		return output

	def _backward(self, gradient):
		for layer in reversed(self.layers):
			gradient = layer.backward(gradient)

	def train(self, X, y):
		#Training the model on single batch size
		y_pred = self._forward(X)
		# print(self.loss_function.loss(y, y_pred).shape)
		loss = np.mean(self.loss_function.loss(y, y_pred))
		# print(loss.shape)
		accuracy = self.loss_function.RMSE(y,y_pred)

		gradient = self.loss_function.derivative(y, y_pred)
		self._backward(gradient = gradient)

		return loss, accuracy

	def test(self, X, y):
		'''Testing model in the single batch of training process'''
		y_pred = self._forward(X)
		loss = np.mean(self.loss_function.loss(y, y_pred))
		accuracy = self.loss_function.RMSE(y, y_pred)

		return loss, accuracy

	def fit(self, X, y, n_epochs, batch_size):
		#Method to fit neural networks with the data samples.
		for i in range(n_epochs):
			self.iterations += 1
			# print(self.iterations)
			batch_error = []
			for X_batch, y_batch in batch_iterator(X, y, batch_size = batch_size):
				loss, acc = self.train(X_batch, y_batch)

				batch_error.append(loss)
			self.errors["training"].append(np.mean(batch_error))

			if self.validation_set is not None:
				val_loss, acc = self.test(self.validation_set["X"], self.validation_set["y"])
				self.errors["validation"].append(val_loss)
		return self.errors["training"], self.errors["validation"]

	def predict(self, X):
		return self._forward(X)



