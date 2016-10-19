import random
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

xes = []
costs = []

# gradient descent for x^2 + y^2 + 2sin(2pix) + 2sin(2piy)
x_t = 0.1
y_t = 0.1
alpha = 0.1
numIterations = 50
cost = 1
for i in range(0, numIterations):
	cost = x_t ** 2 + y_t ** 2 + 2 * np.sin(2 * np.pi * x_t) + 2 * np.sin(2 * np.pi * y_t)
	x_t -= alpha * (2 * x_t + 2 * np.pi * np.cos(2 * np.pi * x_t))
	y_t -= alpha * (2 * y_t + 2 * np.pi * np.cos(2 * np.pi * y_t))
	# cost = np.sum(np.square(loss)) / (2 * size)
	print("Iteration %d | Cost: %f" % (i, cost))
	costs.append(cost)
	xes.append(i)
	# avg gradient per example
	# gradient = np.dot(xTrans, loss) / size
	# update
	# weights = weights - alpha * gradient
	# past_cost = cost
print x_t, y_t

g, = plt.plot(xes, costs, 'bo', label='f(x)')
plt.xlabel('Iterations')
plt.ylabel('f(x)')
print cost
# plt.axis([0, 50, 0, ])
plt.legend(handles=[g])
plt.show()