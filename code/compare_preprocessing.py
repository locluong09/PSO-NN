import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from utils.data_preprocessing import min_max_scaler, normalize_scaler
from utils.model_selection import train_test_split

from deep_learning.models import Neural_Networks
from deep_learning.layers import Dense, Activation
from deep_learning.optimizers import (GradientDescent, StochasticGradientDescent,
									 Adagrad, Adadelta, RMSProp, Adam, Adamax)
from deep_learning.loss_functions import SquareLoss

from utils.data_manipulation import R_square, mean_squared_error

from pso.utils import Objective_function
from pso.particle_swarm_optimization import PSO

from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler, StandardScaler


# from keras.optimizers import Adam, Nadam, Adamax
def get_data(file_name):
	dirname = 'data'
	path = os.path.join(dirname, file_name)
	data = pd.read_csv(path)
	return data

def preprocessing(data):
	column_train = ["ON_STREAM_HRS", "AVG_DOWNHOLE_PRESSURE", "AVG_DOWNHOLE_TEMPERATURE",
	"AVG_CHOKE_SIZE_P", "AVG_WHP_P", "AVG_WHT_P"]
	target = ["BORE_OIL_VOL"]
	X = data[column_train].to_numpy()
	y = data[target].to_numpy()
	return X, y

def log10(X):
	return np.log10(X)

def get_best_cases(production, model):
	'''Return the index of best cases'''
	X = production
	y = model
	mat = y - X.reshape(1,-1)
	MSE_mat = np.sum(np.power(mat,2), axis = 1)
	RMSE_mat = np.sqrt(MSE_mat)
	return np.argsort(RMSE_mat)

def main():
	#load data
	
	field_data = get_data("field_data.csv")
	reservoir_parameters = get_data("reservoir_parameters.csv")
	simulation_cases = get_data("simulation_cases.csv")

	# print(reservoir_parameters['Case 187'])
	
	field_data = np.array(field_data)
	production = field_data[:,1].reshape(-1,1)

	X = np.array(reservoir_parameters)[:,1:188]

	y = np.array(simulation_cases)[:,1:188]

	X = X.T
	X = X[:,1:15]
	y = y.T

	best_cases_idx = get_best_cases(production, y)[:50]
	
	# print(X[1,:])
	# print(X.shape)
	upper_bounds = np.max(X, axis = 0)
	
	# upper_bound[0] += 30
	lower_bounds = np.min(X, axis = 0)

	# lower_bounds[0] -= 20
	# upper_bounds = np.max(X[best_cases_idx], axis = 0)
	# lower_bounds = np.min(X[best_cases_idx], axis = 0)
	# print(lower_bounds)
	# print(best_cases)
	# print(X.shape)
	# print(y.shape)
	scaler1 = MinMaxScaler()
	y = scaler1.fit_transform(y)



	X_train, X_test, y_train, y_test = train_test_split(X[best_cases_idx], y[best_cases_idx], 0.1, shuffle = True, seed = 10)
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 0.1, shuffle = True, seed = 10)
	validation_data = (X_val, y_val)
	n_samples, n_features = X.shape
	n_outputs = y.shape[1]

	# X = min_max_scaler(X)


	GD = GradientDescent(0.001)
	SGD = StochasticGradientDescent(learning_rate = 0.001, momentum = 0.9, nesterov = False)
	SGD_nes = StochasticGradientDescent(learning_rate = 0.001, momentum = 0.9, nesterov = True)
	

	model = Neural_Networks(optimizer = SGD_nes,
							loss = SquareLoss,
							validation_data = validation_data)
	model.add(Dense(50, input_shape=(n_features,)))
	model.add(Activation('sigmoid'))
	# model.add(Dense(20))
	# model.add(Activation('tanh'))
	model.add(Dense(n_outputs))
	model.add(Activation('linear'))

	train_err, val_err = model.fit(X[best_cases_idx], y[best_cases_idx], n_epochs = 1000, batch_size=8)


	scaler = MinMaxScaler()
	production = scaler.fit_transform(field_data[:,1].reshape(-1,1))
	
	print(train_err[-1], val_err[-1])

	plt.plot(train_err, 'r', label = "training")
	plt.plot(val_err, 'b', label = 'validation')
	plt.xlabel("Iterations")
	plt.ylabel("Error")
	plt.legend()
	plt.show()

	y_pred = model.predict(X[51,:])
	plt.plot(y_pred.reshape(-1,1), 'r', label = "Prediction")
	plt.plot(y[51,:].reshape(-1,1), 'b', label = "Simulation_cases")
	plt.legend()
	plt.show()

	'''
	Couple neural networks with PSO to find the best combination of
	parameters
	'''
	
	def func(*args):
		x = []
		for i in args:
			x.append(i)
		x = np.asarray(x)
		y_bar = model.predict(x)
		# y_bar = scaler.fit_transform(y_bar.reshape(-1,1))
		return np.linalg.norm(y_bar - production, ord = np.inf)

	objective_class = Objective_function(func, 14,lower_bounds, upper_bounds)
	pso_opt = PSO(nb_generations = 100,
				nb_populations = 100,
				objective_class = objective_class,
				w = 0.1,
				c1 = 1.5,
				c2 = 1.5,
				max_velocity = 20,
				min_velocity = -20)

	best = pso_opt.evolve()
	best_params = np.array(best.position)
	# print(best)
	y_pred = model.predict(best_params)
	print(y_pred)
	prediction = scaler.inverse_transform(y_pred.reshape(-1,1))
	print(prediction)
	field = scaler.inverse_transform(production.reshape(-1,1))



	plt.plot(prediction, 'r', label = "Prediction")
	plt.plot(field, 'k', label = "Actual")
	plt.legend()
	plt.show()

	plt.plot(np.cumsum(prediction),'r', label = "Prediction")
	plt.plot(np.cumsum(field), 'k', label = "Actual")
	plt.legend()
	plt.show()

if __name__ == "__main__":
	main()


