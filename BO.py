import numpy as np
from robo.fmin import bayesian_optimization
from CIFAR.select import bo_train_cifar
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Define the objective_function.
def objective_function(policies, time):
    acc = bo_train_cifar.main(policies, time)
    error = 1 - acc
    return error

# Define the bounds and dimensions of the input space.
lower = np.array([0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0])
upper = np.array([195.99999,1,9,1,9, 195.99999,1,9,1,9, 195.99999,1,9,1,9])

# Start Bayesian optimization to optimize the objective function
for r in range(1, 2):
    results = bayesian_optimization(objective_function, lower, upper, num_iterations=100, i=r)
