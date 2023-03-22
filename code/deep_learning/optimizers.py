import numpy as np
import copy

class GradientDescent():
	def __init__(self, learning_rate = 0.01):
		self.learning_rate = learning_rate
		self.weight_update = None

	def get_update(self, weight, gradient):
		if self.weight_update is None:
			self.weight_update = np.zeros(np.shape(weight))

		self.weight_update = gradient
		return weight - self.learning_rate*self.weight_update


class StochasticGradientDescent():
	def __init__(self, learning_rate = 0.01, momentum = 0.9, nesterov = False):
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.nesterov = nesterov
		self.weight_update = None


	def get_update(self, weight, gradient):
		if self.weight_update is None:
			self.weight_update = np.zeros(np.shape(weight))

		self.weight_update = self.momentum*self.weight_update + self.learning_rate*gradient
		self.velocity = self.weight_update
		if self.nesterov:
			return weight - (self.momentum*self.velocity + self.learning_rate*self.weight_update)
		else:
			return weight - self.weight_update

class Adagrad():
	def __init__(self, learning_rate = 0.01, epsilon = 1e-6):
		self.learning_rate = learning_rate
		self.epsilon = epsilon
		self.accumulator = None
	def get_update(self, weight, gradient):
		if self.accumulator is None:
			self.accumulator = np.zeros(np.shape(weight))
		self.accumulator += np.square(gradient)
		return weight - self.learning_rate*gradient/(np.sqrt(self.accumulator + self.epsilon))

class Adadelta():
	def __init__(self, rho = 0.9, epsilon = 1e-6):
		self.rho = rho
		self.epsilon = epsilon
		self.accumulator = None
		self.accumulator_delta = None

	def get_update(self, weight, gradient):
		if self.accumulator is None:
			self.accumulator = np.zeros(np.shape(weight))
		if self.accumulator_delta is None:
			self.accumulator_delta = np.zeros(np.shape(weight))

		self.accumulator = self.rho * self.accumulator + (1 - self.rho)*np.power(gradient, 2)
		lr = np.sqrt(self.accumulator_delta + self.epsilon) / np.sqrt(self.accumulator + self.epsilon)
		self.weight_update = lr*gradient
		self.accumulator_delta = self.rho * self.accumulator_delta + (1 - self.rho)*(np.power(self.weight_update, 2))

		return weight - self.weight_update

class RMSProp():
	def __init__(self, learning_rate = 0.01, rho = 0.9, epsilon = 1e-6):
		self.learning_rate = learning_rate
		self.rho = rho
		self.epsilon = epsilon
		self.accumulator = None

	def get_update(self, weight, gradient):
		if self.accumulator is None:
			self.accumulator = np.zeros(np.shape(weight))

		self.accumulator = self.rho * self.accumulator + (1 - self.rho) * np.power(gradient, 2)
		return weight - self.learning_rate * gradient / np.sqrt(self.accumulator + self.epsilon)


class Adam():
	def __init__(self, learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8):
		self.learning_rate = learning_rate
		self.beta_1 = beta_1
		self.beta_2 = beta_2
		self.epsilon = epsilon
		self.time_step = 0
		self.ms = None
		self.vs = None
	def get_update(self, weight, gradient):
		if self.ms is None:
			self.ms = np.zeros(np.shape(weight))

		if self.vs is None:
			self.vs = np.zeros(np.shape(weight))

		self.time_step += 1

		self.ms = self.beta_1 * self.ms + (1- self.beta_1) * gradient
		self.vs = self.beta_2 * self.vs + (1- self.beta_2) * np.power(gradient, 2)
		# print(self.time_step)
		# self.learning_rate = self.learning_rate*np.sqrt(1 - self.beta_2** self.time_step) / (1 - self.beta_1**self.time_step)
		# print(self.learning_rate)
		
		return weight - self.learning_rate * self.ms / (np.sqrt(self.vs)+ self.epsilon)

class Adamax():
	def __init__(self, learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8):
		self.learning_rate = learning_rate
		self.beta_1 = beta_1
		self.beta_2 = beta_2
		self.epsilon = epsilon
		self.time_step = 0
		self.vt = None
		self.ut = None

	def get_update(self, weight, gradient):
		if self.ms is None:
			self.vt = np.zeros(np.shape(weight))

		if self.vs is None:
			self.ut = np.zeros(np.shape(weight))

		self.time_step += 1

		self.vt = self.beta_1 * self.vt + (1- self.beta_1) * gradient
		self.ut = np.maximum(self.beta_2 * self.ut, np.abs(gradient))

		# self.learning_rate = self.learning_rate*np.sqrt(1 - self.beta_2** self.time_step) / (1 - self.beta_1**self.time_step)
		# print(self.learning_rate)
		
		return weight - self.learning_rate * self.vt/ (self.ut+ self.epsilon)

class NAdam():
	def __init__(self, learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8):
		self.learning_rate = learning_rate
		self.beta_1 = beta_1
		self.beta_2 = beta_2
		self.epsilon = epsilon
		self.time_step = 0
		self.mt = None
		self.vt = None

	def get_update(self, weight, gradient):
		if self.ms is None:
			self.mt = np.zeros(np.shape(weight))

		if self.vs is None:
			self.vt = np.zeros(np.shape(weight))

		self.mt = self.beta_1 * self.mt + (1- self.beta_1) * gradient
		self.mt_bar = self.mt/(1 - self.beta_1)

		self.vt = self.beta_2 * self.vt + (1 - self.beta_2) * np.power(gradient, 2)
		self.vt_bar = self.vt/(1 - self.beta_2)

		# self.learning_rate = self.learning_rate*np.sqrt(1 - self.beta_2** self.time_step) / (1 - self.beta_1**self.time_step)
		# print(self.learning_rate)
		
		return weight - self.learning_rate * (self.mt_bar * self.beta_1 + (1 - self.beta_1)/self.beta_1 *gradient)/ (np.sqrt(self.vt_bar)+ self.epsilon)

class PSO():
	def __init__(self, model,nb_generations, nb_populations, w, c1, c2, max_velocity, min_velocity):
		self.model = model
		self.nb_generations = nb_generations
		self.nb_populations = nb_populations
		self.w = w
		self.c1 = c1
		self.c2 = c2
		self.max_velocity = max_velocity
		self.min_velocity = min_velocity
		self.swarm = []
		self.gbest = None

	def make_particle(self):
		particle = self.model(self.X.shape[1])
		particle.fitness = float('inf')
		particle.best_fitness = float('inf')
		particle.RMSE = 0
		particle.pbest = copy.copy(particle.layers)
		particle.velocity = []
		for layer in particle.layers:
			if hasattr(layer, 'weight'):
				velocity = {'weight': np.zeros_like(layer.weight), 'bias': np.zeros_like(layer.bias)}
			particle.velocity.append(velocity)

		return particle

	def set_swarm(self):
		for i in range(self.nb_populations):
			particle = self.make_particle()
			self.swarm.append(particle)

	def update_velocity(self, particle):
		r1 = np.random.uniform()
		r2 = np.random.uniform()

		for i, layer in enumerate(particle.layers):
			if hasattr(layer, 'weight'):
				inertia_term = self.w * particle.velocity[i]['weight']
				cognitive_term = r1 * self.c1 * (particle.pbest[i].weight - layer.weight)
				social_term = r2 * self.c2 * (self.gbest.layers[i].weight - layer.weight)
				total = inertia_term + cognitive_term + social_term
				particle.velocity[i]['weight'] = np.clip(total, self.min_velocity, self.max_velocity)


				inertia_term_ = self.w * particle.velocity[i]['bias']
				cognitive_term_ = r1 * self.c1 * (particle.pbest[i].bias - layer.bias)
				social_term_ = r2 * self.c2 * (self.gbest.layers[i].bias - layer.bias)
				total_ = inertia_term_ + cognitive_term_ + social_term_
				particle.velocity[i]['bias'] = np.clip(total_, self.min_velocity, self.max_velocity)

				particle.layers[i].weight += particle.velocity[i]['weight']
				particle.layers[i].bias += particle.velocity[i]['bias']


	def calculate_fitness(self, particle):
		loss, RMSE  = particle.test(self.X, self.y)
		particle.fitness = loss
		particle.RMSE = RMSE


	def evolve(self, X, y):
		self.X = X
		self.y = y
		#create swarm
		self.set_swarm()
		#set gbest
		self.gbest = copy.copy(self.swarm[0])

		for i in range(self.nb_generations):
			for particle in self.swarm:
				self.update_velocity(particle)
				self.calculate_fitness(particle)

				if particle.fitness < particle.best_fitness:
					particle.best_fitness = particle.fitness
					particle.pbest = copy.copy(particle.layers)

				if particle.fitness < self.gbest.fitness:
					self.gbest = copy.copy(particle)

			print("At iteration level : {}, best fitness : {}, and RMSE : {}".format(i+1, self.gbest.fitness, self.gbest.RMSE))

		return self.gbest





















