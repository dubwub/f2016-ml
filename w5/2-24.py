import math
import numpy as np
import matplotlib.pyplot as plt
import random

def generatePoint():
	x0 = random.uniform(-1, 1)
	y0 = x0 * x0
	x1 = random.uniform(-1, 1)
	y1 = x1 * x1
	slope = (y1 - y0)/(x1 - x0)
	yint = y1 - slope*x1
	return [slope, yint]

def runExperiment():
	results = []
	gbarslope = 0
	gbaryint = 0
	for i in range(0, 1000):
		newPoint = generatePoint()
		gbarslope = gbarslope + newPoint[0]
		gbaryint = gbaryint + newPoint[1]
		results.append(generatePoint())
	gbarslope /= 1000
	gbaryint /= 1000
	print "gbarslope, gbaryint", gbarslope, gbaryint
	bias = 0
	var = 0
	x = -1;
	stepsize = 0.001
	while x <= 1:
		actual_val = math.pow(x,2)
		gbar_val = (gbarslope)*x + gbaryint
		bias += math.pow(gbar_val - actual_val, 2)
		for i in range(0, 1000):
			result_val = results[i][0]*x + results[i][1]
			var += math.pow(result_val - gbar_val, 2) / (2000/(stepsize))
		x += 0.001
	bias /= (2/stepsize)
	print "bias:", bias
	print "variance:", var
	return [gbarslope, gbaryint]

results = runExperiment()
x = np.arange(-1, 1, 0.01)
g_line, = plt.plot(x, x*results[0] + results[1], 'k', label='gbar')
f_line, = plt.plot(x, x*x, 'r', label='f')
plt.legend(handles=[f_line, g_line])
plt.show()