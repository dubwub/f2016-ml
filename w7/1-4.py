import random
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
f = open('ZipDigits.train', 'r')
# f = open('ZipDigits.test', 'r')

ones_x = []
ones_y = []
fives_x = []
fives_y = []
x_matrix = []
y_matrix = []

# calculate symmetry about y-axis and x-axis
def getSymmetry(array):
	val = 0
	for i in range(0,8):
		for j in range(0, 16):
			val += abs(float(array[j*16 + i]) - float(array[(j+1)*16 - (i + 1)]))
	for i in range(0,16):
		for j in range(0,8):
			val += abs(float(array[j*16 + i]) - float(array[(15 - j)*16 - i]))
	val /= 128
	return val

def getIntensity(array):
	val = 0
	for i in range(0,256):
		# val += np.power(float(array[i]) + 1, 2)
		val += float(array[i]) + 1
	return val / 256

max_symmetry = 0
max_intensity = 0
for line in f:
	line = line.split(' ')
	if line[0] == '1.0000':
		ones_x.append(getSymmetry(line[1:-1]))
		if ones_x[len(ones_x) - 1] > max_symmetry:
			max_symmetry = ones_x[len(ones_x) - 1]
		ones_y.append(getIntensity(line[1:-1]))
		if ones_y[len(ones_x) - 1] > max_intensity:
			max_intensity = ones_y[len(ones_y) - 1]
	elif line[0] == '5.0000':
		fives_x.append(getSymmetry(line[1:-1]))
		if fives_x[len(fives_x) - 1] > max_symmetry:
			max_symmetry = fives_x[len(fives_x) - 1]
		fives_y.append(getIntensity(line[1:-1]))
		if fives_y[len(fives_x) - 1] > max_intensity:
			max_intensity = fives_y[len(fives_y) - 1]

for i in range(0, len(ones_x)):
	ones_x[i] /= max_symmetry
	# ones_x[i] = (ones_x[i])
	ones_y[i] /= max_intensity
	# ones_y[i] = (ones_y[i] - .5) ** 3
	# 
	x_matrix.append([1, ones_x[i], ones_y[i], ones_x[i] ** 2, ones_x[i] * ones_y[i], ones_y[i] ** 2, ones_x[i] ** 3, ones_x[i] ** 2 * ones_y[i],
		ones_y[i] ** 2 * ones_x[i], ones_y[i] **3])
	y_matrix.append([1])
for i in range(0, len(fives_x)):
	fives_x[i] /= max_symmetry
	# fives_x[i] = (fives_x[i] - .5) ** 3
	fives_y[i] /= max_intensity
	# fives_y[i] = (fives_y[i] - .5) ** 3
	x_matrix.append([1, fives_x[i], fives_y[i], fives_x[i] ** 2, fives_x[i] * fives_y[i], fives_y[i] ** 2, fives_x[i] ** 3, fives_x[i] ** 2 * fives_y[i],
		fives_y[i] ** 2 * fives_x[i], fives_y[i] **3])
	y_matrix.append([-1])

# print len(ones_x) + len(fives_x)

ones, = plt.plot(ones_x, ones_y, 'bo', label='1')
fives, = plt.plot(fives_x, fives_y, 'rx', label='5')

# logistic regression w/ gradient descent
weights = []
for i in range(0, 10):
	weights.append(np.random.normal(loc=0.0, scale=.0001))
weights = np.matrix(weights).transpose()
x_matrix = np.matrix(x_matrix)
y_matrix = np.matrix(y_matrix)
xTrans = x_matrix.transpose()
alpha = 0.1
numIterations = 100000
termination_Ein = 0.05
termination_delta = .0001
past_cost = 1
size = len(ones_x) + len(fives_x)
for i in range(0, numIterations):
	hypothesis = np.dot(x_matrix, weights)
	# print hypothesis
	loss = hypothesis - y_matrix
	cost = np.sum(np.square(loss)) / (2 * size)
	# print("Iteration %d | Cost: %f" % (i, cost))
	# avg gradient per example
	gradient = np.dot(xTrans, loss) / size
	# update
	weights = weights - alpha * gradient
	if past_cost - cost < termination_delta and cost < termination_Ein:
		break
	past_cost = cost
print weights
print past_cost

x = 0
y = 0
g_xes = []
g_yes = []
while x < 1:
	while y < 1:
		# print abs(weights[0] + weights[1]*x + weights[2]*y + weights[3]*(x ** 2) + weights[4]*x*y + weights[5]*(y**2) + weights[6]*(x**3) + weights[7]*(x**2)*y
		# 	+ weights[8]*(y**2)*x + weights[9]*(y**3))
		if abs(weights[0] + weights[1]*x + weights[2]*y + weights[3]*(x ** 2) + weights[4]*x*y + weights[5]*(y**2) + weights[6]*(x**3) + weights[7]*(x**2)*y
			+ weights[8]*(y**2)*x + weights[9]*(y**3)) < .01:
			g_xes.append(x)
			g_yes.append(y)
		y += .01
	x += .01
	y = 0
g, = plt.plot(g_xes, g_yes, 'go', label='g')

# w0 +  
# g_slope = -weights[1]/weights[2]
# g_intercept = -weights[0]/weights[2]
# g_slope = g_slope.item(0)
# g_intercept = g_intercept.item(0)
# print g_slope, g_intercept
# x = np.arange(-1, 1, 0.01)
# g, = plt.plot(x, , 'g', label='g')
plt.xlabel('Symmetry')
plt.ylabel('Intensity')
plt.axis([0, 1, 0, 1])
plt.legend(handles=[ones, fives, g])
plt.show()