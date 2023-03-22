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
	print(mat.shape)
	MSE_mat = np.sum(np.power(mat,2), axis = 1)
	RMSE_mat = np.sqrt(MSE_mat)
	return np.argsort(RMSE_mat), RMSE_mat

def main():
	#load data
	#Data manipulation
	#--------------------------------------
	field_data = get_data("field_data.csv")
	reservoir_parameters = get_data("reservoir_parameters.csv")
	simulation_cases = get_data("simulation_cases.csv")

	
	field_data = np.array(field_data)
	production = log10(field_data[:,1]).reshape(-1,1)

	X = np.array(reservoir_parameters)[:,1:188]

	y = np.array(simulation_cases)[:,1:188]

	X = X.T
	y = y.T
	y = log10(y)
	#--------------------------------------
	
	'''Return best cases index for training neural networks'''
	best_cases_idx, RMSE = get_best_cases(production, y)

	'''Return the uppper and lower bounds of parameters'''
	upper_bounds = np.max(X, axis = 0)
	lower_bounds = np.min(X, axis = 0)

	'''Train test split'''
	X_train, X_test, y_train, y_test = train_test_split(X[best_cases_idx], y[best_cases_idx], 0.1, shuffle = True, seed = 10)
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 0.1, shuffle = True, seed = 10)
	validation_data = (X_val, y_val)
	n_samples, n_features = X.shape
	n_outputs = y.shape[1]



	GD = GradientDescent(0.001)
	SGD = StochasticGradientDescent(learning_rate = 0.001, momentum = 0.9, nesterov = False)
	SGD_nes = StochasticGradientDescent(learning_rate = 0.001, momentum = 0.9, nesterov = True)
	Ada = Adagrad(learning_rate = 0.001, epsilon = 1e-6)
	Adad = Adadelta(rho = 0.9, epsilon = 1e-6)
	RMS = RMSProp(learning_rate = 0.01)
	Adam_opt = Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-6)
	Adamax_opt = Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-6)
	NAdam_opt = Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-6)
	NAdam_opt = Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-6)
	
	#Define neural networks model
	model = Neural_Networks(optimizer = SGD_nes,
							loss = SquareLoss,
							validation_data = validation_data)
	model.add(Dense(100, input_shape=(n_features,)))
	model.add(Activation('sigmoid'))
	model.add(Dense(n_outputs))
	model.add(Activation('linear'))


	train_err, val_err = model.fit(X_train, y_train, n_epochs = 1000, batch_size=8)
	field_log = log10(field_data[:,1])
	
	print("Traing error is {}, validation error is {}".format(train_err[-1], val_err[-1]))
	plt.plot(train_err, 'r', label = "training")
	plt.plot(val_err, 'b', label = 'validation')
	plt.xlabel("Iterations")
	plt.ylabel("Error")
	plt.legend()
	plt.show()

	y_pred = model.predict(X[186,:])
	plt.plot(y_pred.reshape(-1,1), 'r', label = "Prediction")
	plt.plot(y[186,:].reshape(-1,1), 'b', label = "Simulation_cases 187")
	plt.legend()
	plt.show()




if __name__ == "__main__":
	main()