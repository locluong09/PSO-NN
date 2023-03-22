import numpy as np
import math
import copy

from deep_learning.initializations import constant_value, xavier_uniform
from deep_learning.activations import Linear, Sigmoid, Tanh, LeakyRelu, Softmax, Relu

class Dense():
	def __init__(self, nb_units, input_shape = None):
		self.input_shape = input_shape
		self.nb_units = nb_units

	def set_input_shape(self, shape):
		self.input_shape = shape

	def initialize(self):
		self.weight = xavier_uniform(shape = (self.input_shape[0], self.nb_units))
		self.bias = constant_value(0, shape = (self.nb_units,))

	def set_optimizer(self, optimizer):
		self.weight_optimizer = copy.copy(optimizer)
		self.bias_optimizer = copy.copy(optimizer)

	def forward(self, _input):
		self.layer_input = _input
		return _input.dot(self.weight) + self.bias

	def backward(self, gradient):
		grad_weight = self.layer_input.T.dot(gradient)
		grad_bias = np.sum(gradient, axis = 0)
		self.weight = self.weight_optimizer.get_update(self.weight, grad_weight)
		self.bias = self.bias_optimizer.get_update(self.bias, grad_bias)
		# Update gradient 
		gradient = gradient.dot(self.weight.T)
		return gradient

	def output_shape(self):
		return (self.nb_units, )


activation_functions = {
	'relu': Relu,
	'sigmoid': Sigmoid,
	'leaky_relu': LeakyRelu,
	'tanh': Tanh,
	'linear': Linear,
	'softmax': Softmax
}

class Activation():
	def __init__(self, activation_name):
		self.activation_class = activation_functions[activation_name]()

	def set_input_shape(self, shape):
		self.input_shape = shape

	def forward(self, _input):
		self.layer_input = _input
		return self.activation_class.activate(_input)

	def backward(self, gradient):
		return gradient * self.activation_class.derivative(self.layer_input)

	def output_shape(self):
		return self.input_shape

