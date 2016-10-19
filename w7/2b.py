import random
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

# gradient descent for x^2 + y^2 + 2sin(2pix) + 2sin(2piy)
def runExperiment(x_t, y_t, alpha):
	# x_t = 0.1
	# y_t = 0.1
	# alpha = 0.1
	numIterations = 50
	cost = 1
	for i in range(0, numIterations):
		cost = x_t ** 2 + y_t ** 2 + 2 * np.sin(2 * np.pi * x_t) + 2 * np.sin(2 * np.pi * y_t)
		x_t -= alpha * (2 * x_t + 2 * np.pi * np.cos(2 * np.pi * x_t))
		y_t -= alpha * (2 * y_t + 2 * np.pi * np.cos(2 * np.pi * y_t))
	print x_t, y_t, cost

runExperiment(0.1, 0.1, 0.01)
runExperiment(0.1, 0.1, 0.1)
runExperiment(1, 1, 0.01)
runExperiment(1, 1, 0.1)
runExperiment(-0.5, -0.5, 0.01)
runExperiment(-0.5, -0.5, 0.1)
runExperiment(-1, -1, 0.01)
runExperiment(-1, -1, 0.1)