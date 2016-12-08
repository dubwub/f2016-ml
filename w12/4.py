from sklearn import svm
import random
import numpy as np
import matplotlib.pyplot as plt
f = open('training.txt', 'r')

hidden = 10 # number of hidden pts

# reading in points

x_matrix = []
y_matrix = []

pos_x = []
pos_y = []
neg_x = []
neg_y = []

for line in f:
	split = line.split(' ')
	x_matrix.append([np.float32(split[1]), np.float32(split[2])])
	if split[0] == '1':
		y_matrix.append(1)
		pos_x.append(np.float32(split[1]))
		pos_y.append(np.float32(split[2]))
	else:
		y_matrix.append(-1)
		neg_x.append(np.float32(split[1]))
		neg_y.append(np.float32(split[2]))

clf = svm.SVC(C=25.0)
clf.fit(x_matrix, y_matrix)

p_x = []
p_y = []
n_x = []
n_y = []

# print clf.predict([-1, -1]).item(0)
# SHOWING DECISION BOUNDS
x_it = -1
y_it = -1
while x_it <= 1:
	predict = clf.predict([x_it, y_it])
	# print predict
	if predict.item(0) > 0:
		p_x.append(x_it)
		p_y.append(y_it)
	else:
		n_x.append(x_it)
		n_y.append(y_it)
	y_it += .02
	if y_it >= 1:
		y_it = -1
		x_it += .02

plt.xlabel('symmetry')
plt.ylabel('intensity')
blues, = plt.plot(p_x, p_y, 'bx')
reds, = plt.plot(n_x, n_y, 'rx')
ones, = plt.plot(pos_x, pos_y, 'bo', label='1')
other, = plt.plot(neg_x, neg_y, 'ro', label='not 1')
plt.show()
