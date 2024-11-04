# Coupling PSO and NN for History Matching

## Overview
This project combines **Particle Swarm Optimization (PSO)** with **Neural Networks (NN)** for efficient and accurate history matching. History matching is a method used to calibrate model parameters by minimizing the difference between simulated and observed data. By leveraging the optimization capabilities of PSO and the predictive power of neural networks, this approach aims to improve accuracy and computational efficiency in history matching tasks.

## Methodology
- **Particle Swarm Optimization (PSO)**: PSO is a nature-inspired optimization algorithm based on the social behavior of birds. It searches for optimal solutions by updating particles (potential solutions) based on their own experience and the experience of their neighbors.
- **Neural Network (NN)**: The neural network is trained to approximate the objective function and guide PSO in searching the parameter space, reducing the need for full-scale simulations.

## Features
- Combines PSO with NN for parameter optimization in history matching.
- Reduces computational cost by minizing objective functions.

## TODO
- Supports various neural network architectures and PSO configurations.

## Dependencies
This project requires the following Python packages:
- **NumPy**: For numerical operations.
- **Matplotlib**: For data visualization.
- **Pandas**: For data manipulation and analysis.

## Installation
Clone the repository:
```bash
git https://github.com/locluong09/PSO-NN
cd PSO-NN
cd code
python case.py
