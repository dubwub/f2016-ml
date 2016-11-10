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

# get Z

# part 3
# wreg = np.linalg.inv(np.transpose(x_matrix) * x_matrix) * np.transpose(x_matrix) * y_matrix

# part 4 
# wreg = np.linalg.inv(np.transpose(x_matrix) * x_matrix + 2 * np.identity(45)) * np.transpose(x_matrix) * y_matrix
#print wreg

# gx = []
# gy = []
# gxit = -1
# gyit = -1
# while gxit < 1:
# 	output = np.transpose(wreg) * np.transpose(np.matrix(legendreTransform(np.float32(gxit), np.float32(gyit))))
# 	# print wreg.shape
# 	# print np.matrix(legendreTransform(np.float32(gxit), np.float32(gyit))).shape
# 	# print output
# 	if output < .05 and output > -.05:
# 		gx.append(gxit)
# 		gy.append(gyit)
# 	gyit += .02
# 	if gyit > 1:
# 		gyit = -1
# 		gxit += .02

# g, = plt.plot(gx, gy, 'ro', label='g')
# ones, = plt.plot(one_x, one_y, 'bo', label='1')
# other, = plt.plot(other_x, other_y, 'rx', label='not 1')

# plt.xlabel('Symmetry')
# plt.ylabel('Intensity')
# plt.axis([-1, 1, -1, 1])
# # plt.legend(handles=[ones, other, g])
# plt.legend(handles=[ones, other])
# plt.show()

# part 4b (E_test vs wreg)
reg = 0
reg_x = []
reg_y = []
reg_ecv = []
while reg < 2:
	wreg = np.linalg.inv(np.transpose(x_matrix) * x_matrix + reg * np.identity(45)) * np.transpose(x_matrix) * y_matrix
	reg_x.append(reg)
	guess = np.transpose(wreg) * np.transpose(x_matrix)
	guess = np.transpose(guess)
	error = y_matrix - guess
	# print error
	error_val = np.transpose(error) * error
	reg_y.append(np.sum(error_val/300))

	h_mat = x_matrix * np.linalg.inv(np.transpose(x_matrix) * x_matrix + reg * np.identity(45)) * np.transpose(x_matrix)
	guess2 = h_mat * x_matrix
	print h_mat
	# print guess2
	# print h_mat.shape
	ecv = 0
	for i in range(0, 300):
		add_to_ecv = y_matrix[i] - guess2[i]
		add_to_ecv /= (1 - h_mat.item((i,i)))
		add_to_ecv = np.power(add_to_ecv, 2)
		# print add_to_ecv
		ecv += add_to_ecv
	ecv /= 300
	# print ecv
	reg_ecv.append(np.sum(ecv))
	reg += .002
plt.xlabel('Regularization')
plt.ylabel('Error')
ecv, = plt.plot(reg_x, reg_ecv, 'bo', label='Ecv(lambda)')
# etest, = plt.plot(reg_x, reg_y, 'ro', label='Etest(wreg(lambda))')
# plt.axis([0, 2, 0, 1])
plt.legend(handles=[ecv])
plt.show()