import numpy as np
import matplotlib.pyplot as plt
f = open('points.txt', 'r')

one_x = []
one_y = []
other_x = []
other_y = []
x_matrix = []
y_matrix = []

def legendre(value, n):
	if n == 0:
		return 1
	elif n == 1:
		return value
	else:
		return (2*n - 1)/(n) * value * legendre(value, n - 1) - (n - 1)/(n) * legendre(value, n - 2)

def legendreTransform(v1, v2):
	return_val = [1]
	for x in range(1, 9):
		v1_exp = x
		while v1_exp >= 0:
			return_val.append(legendre(v1, v1_exp) * legendre(v2, x - v1_exp))
			v1_exp -= 1
	return return_val

for line in f:
	split = line.split(' ')
	x_matrix.append(legendreTransform(np.float32(split[1]), np.float32(split[2])))
	# print len(x_matrix[0])
	# print x_matrix
	if split[0] == '1':
		y_matrix.append([1])
		one_x.append(np.float32(split[1]))
		one_y.append(np.float32(split[2]))
	else:
		y_matrix.append([-1])
		other_x.append(np.float32(split[1]))
		other_y.append(np.float32(split[2]))

# print x_matrix
x_matrix = np.matrix(x_matrix)
y_matrix = np.matrix(y_matrix)

ones, = plt.plot(one_x, one_y, 'bo', label='1')
other, = plt.plot(other_x, other_y, 'rx', label='not 1')

# get Z
wreg = np.linalg.inv(np.transpose(x_matrix) * x_matrix) * np.transpose(x_matrix) * y_matrix
print wreg

# gx = []
# gy = []
# gxit = -1
# gyit = -1
# while gxit < 1:
# 	output = np.transpose(wreg) * np.transpose(np.matrix(legendreTransform(np.float32(gxit), np.float32(gyit))))
# 	# print wreg.shape
# 	# print np.matrix(legendreTransform(np.float32(gxit), np.float32(gyit))).shape
# 	# print output
# 	if output < .2 and output > -.2:
# 		gx.append(gxit)
# 		gy.append(gyit)
# 	gyit += .02
# 	if gyit > 1:
# 		gyit = -1
# 		gxit += .02

# g, = plt.plot(gx, gy, 'ro', label='g')

plt.xlabel('Symmetry')
plt.ylabel('Intensity')
plt.axis([-1, 1, -1, 1])
# plt.legend(handles=[ones, other, g])
plt.legend(handles=[ones, other])
plt.show()