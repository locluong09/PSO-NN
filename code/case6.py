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
	MSE_mat = np.sum(np.power(mat,2), axis = 1)
	RMSE_mat = np.sqrt(MSE_mat)
	return np.argsort(RMSE_mat)

def get_best(production, model):
	X = production
	y = model
	mat = y - X.reshape(1,-1)
	mat = np.linalg.norm(mat, ord = np.inf, axis = 1)
	return np.argsort(mat)

def get_best_new(production, model):
	X = production
	y = model
	max_pro = np.max(production)
	max_mat = np.max(y,axis = 1)
	maximum = abs(max_mat - max_pro)
	return np.argsort(maximum)



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

	best_cases_idx1 = list(get_best_new(production, y)[:30])
	best_cases_idx2 = list(get_best(production, y)[:20])
	best_cases_idx3 = list(get_best_cases(production, y)[:20])


	best_cases = best_cases_idx1+best_cases_idx2+best_cases_idx3
	best_cases = np.unique(best_cases)
	print(best_cases)
	
	
	upper_bounds = np.max(X, axis = 0)
	lower_bounds = np.min(X, axis = 0)
	

	X_train, X_test, y_train, y_test = train_test_split(X[best_cases], y[best_cases], 0.1, shuffle = True, seed = 10)
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 0.1, shuffle = True, seed = 10)
	validation_data = (X_val, y_val)
	n_samples, n_features = X.shape
	n_outputs = y.shape[1]

	# X = min_max_scaler(X)


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
	
	model = Neural_Networks(optimizer = SGD_nes,
							loss = SquareLoss,
							validation_data = validation_data)
	model.add(Dense(100, input_shape=(n_features,)))
	model.add(Activation('sigmoid'))
	model.add(Dense(n_outputs))
	model.add(Activation('linear'))

	train_err, val_err = model.fit(X_train, y_train, n_epochs = 100, batch_size=8)

	field_log = log10(field_data[:,1])
	
	print(train_err[-1], val_err[-1])
	# print(field_log)
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
	Couple neural netwokrs with PSO to find the best combination of
	parameters
	'''
	
	def func(*args):
		x = []
		for i in args:
			x.append(i)
		x = np.asarray(x)
		y_bar = model.predict(x)
		return (0.2*np.linalg.norm(y_bar - field_log, ord = 2) + 0*np.linalg.norm(y_bar - field_log, ord = 1)+ 0.8*np.linalg.norm(y_bar - field_log, ord = np.inf))

	objective_class = Objective_function(func, 15,lower_bounds, upper_bounds)
	pso_opt = PSO(nb_generations = 50,
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
	print(np.linalg.norm(y_pred-field_log, ord = 2))
	prediction = np.power(10,y_pred)
	field = np.power(10,field_log)
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


