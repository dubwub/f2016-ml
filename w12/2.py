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
		y_matrix.append([1])
		pos_x.append(np.float32(split[1]))
		pos_y.append(np.float32(split[2]))
	else:
		y_matrix.append([-1])
		neg_x.append(np.float32(split[1]))
		neg_y.append(np.float32(split[2]))

# done reading

w0 = []
w1 = []

def init_weights():
	for i in range(0, 3):
		tmp = []
		for i in range(0, hidden):
			tmp.append(random.uniform(-.25, .25))
		w0.append(tmp)
	for i in range(0, hidden+1):
		w1.append(random.uniform(-.25, .25))

init_weights()
w0 = np.matrix(w0)
w1 = np.matrix(w1)

errors = []
iterations = []

def iterate(w0_, w1_, x_matrix, y_matrix, get_gradient):
	e_in = 0
	gradients = [0 * w0_, 0 * w1_]

	for i in range(len(x_matrix)):
		x0 = np.matrix([1, x_matrix[i][0], x_matrix[i][1]])
		x1 = [1]
		s2 = []
		x2 = 0
		# forward prop
		for j in range(0, hidden): # l = 0 to l = 1
			value = np.dot(x0, w0_[:,j])
			x1.append(value)
		x1 = np.matrix(x1)
		x2 = np.dot(x1, np.transpose(w1_))
		# if (x2.item(0) > 0):
		# 	x2 = 1.0
		# else:
		# 	x2 = -1.0
		e_in += np.power((x2 - y_matrix[i][0]), 2)
		# x2 = x2.item(0)
		e_in = e_in.item(0)

		if get_gradient == True:
			d2 = 2*(x2 - y_matrix[i][0])
			d1 = d2 * w1_[0,1:]
			deW1 = np.dot(np.transpose(x0), np.matrix(d1)) + .1/(len(x_matrix)) * w0_
			gradients[0] += 1.0/len(x_matrix) * deW1
			deW2 = np.dot(np.transpose(x1), np.matrix(d2))
			gradients[1] += 1.0/len(x_matrix) * np.transpose(deW2) + .1/(len(x_matrix)) * w1_
	e_in /= 4.0*len(x_matrix) # 4 * 300
	
	return [e_in, gradients]

alpha = 1.1
beta = .8
step_size = .1
for i in range(0, 1000):
	if i%1000 == 0 and i > 0:
		print i, step_size, errors[len(errors) - 1]
	output = iterate(w0, w1, x_matrix, y_matrix, True)
	e_in = output[0]
	gradients = output[1]
	newe_in = e_in + 1
	while newe_in > e_in and i < 1000:
		i += 1
		neww0 = w0 - step_size * gradients[0]
		neww1 = w1 - step_size * gradients[1]
		new_output = iterate(neww0, neww1, x_matrix, y_matrix, False) 
		newe_in = new_output[0]
		if newe_in > e_in:
			step_size *= beta
	errors.append(e_in)
	iterations.append(i)
	norm1 = np.linalg.norm(gradients[0])
	norm2 = np.linalg.norm(gradients[1])
	if norm1 + norm2 < .0001 and errors[len(errors) - 1] < .02:
		break
	w0 = neww0
	w1 = neww1
	step_size *= alpha

plt.plot(iterations, errors, 'bo', label='E_in')
plt.xlabel('iterations')
plt.ylabel('E_in')
plt.show()

# print errors
print step_size, errors[len(errors) - 1]

p_x = []
p_y = []
n_x = []
n_y = []

# SHOWING DECISION BOUNDS
x_it = -1
y_it = -1
while x_it <= 1:
	# forward propagate
	x0 = [1, x_it, y_it]
	x1 = [1]
	s2 = []
	x2 = 0
	for j in range(0, hidden): # l = 0 to l = 1
		value = 1 * w0.item((0, j)) + x_it * w0.item((1, j)) + y_it*w0.item((2, j))
		x1.append(value)
	final_val = 0
	
	for j in range(0, hidden+1):
		final_val += x1[j] * w1.item(j)
	x2 = final_val
	if x2 > 0:
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