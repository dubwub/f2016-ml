import numpy as np
import matplotlib.pyplot as plt
f = open('points.txt', 'r')
ftest = open('test_pts.txt', 'r')

x_matrix = []
y_matrix = []

# UNCOMMENT THE COMMENTS BELOW TO DISPLAY POINTS.TXT

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

# ones, = plt.plot(pos_x, pos_y, 'bo', label='1')
# other, = plt.plot(neg_x, neg_y, 'rx', label='not 1')
# plt.show()

# END UNCOMMENT BLOCK

# pt1 is [distance, y_val]
def comparator(pt1):
	return pt1[0]

# distance_matrix = []
# for i in range(0, len(x_matrix)):
# 	dists = []
# 	for j in range(0, len(x_matrix)):
# 		# if i == j:
# 		# 	continue
# 		dist = np.power(np.power(x_matrix[i][0] - x_matrix[j][0], 2) + np.power(x_matrix[i][1] - x_matrix[j][1], 2), 0.5)
# 		dists.append([dist, y_matrix[j]])
# 	dists = sorted(dists, key=comparator)
# 	# print dists
# 	distance_matrix.append(dists)

# print distance_matrix

# now that we have all the sorted distances from each point, we can find the optimal k-NN
# min_error = 1
# min_k = 0
# errors = []
# ks = []
# k = 1
# while k < len(x_matrix)/2:
# 	error = 0
# 	for i in range(0, len(y_matrix)): # distance matrix is 300 x 299
# 		sum = 0
# 		for j in range(0, k):
# 			# print distance_matrix[i][j][1][0]
# 			sum += distance_matrix[i][j][1][0]
# 		# print sum, y_matrix[i][0]
# 		if (sum < 0 and y_matrix[i][0] > 0) or (sum > 0 and y_matrix[i][0] < 0):
# 			error += 1.0/len(y_matrix)
# 	# print error, k
# 	errors.append(error)
# 	ks.append(k)
# 	if error < min_error:
# 		min_error = error
# 		min_k = k
# 	k += 2

# print min_error, min_k

# UNCOMMENT BELOW FOR ECV GRAPH
# kgraph, = plt.plot(ks, errors, 'bo', label='k')
# plt.legend(handles=[kgraph])
# plt.xlabel('k-NN')
# plt.ylabel('Ecv')
# plt.show()
# END UNCOMMENT BLOCK

# RUNNING JUST ON 7-NN, TRYING TO FIND E_IN
# error = 0
# k = 7
# for i in range(0, len(y_matrix)): # distance matrix is 300 x 299
# 	sum = 0
# 	for j in range(0, k):
# 		# print distance_matrix[i][j][1][0]
# 		sum += distance_matrix[i][j][1][0]
# 	# print sum, y_matrix[i][0]
# 	if (sum < 0 and y_matrix[i][0] > 0) or (sum > 0 and y_matrix[i][0] < 0):
# 		error += 1.0/len(y_matrix)
# print error, k

# p_x = []
# p_y = []
# n_x = []
# n_y = []

# # SHOWING DECISION BOUNDS
# x_it = -1
# y_it = -1
# while x_it <= 1:
# 	dists = []
# 	for j in range(0, len(x_matrix)):
# 		dist = np.power(np.power(x_it - x_matrix[j][0], 2) + np.power(y_it - x_matrix[j][1], 2), 0.5)
# 		dists.append([dist, y_matrix[j]])
# 	dists = sorted(dists, key=comparator)
# 	sum = 0
# 	for k in range(0, 7):
# 		# print dists[k][1][0]
# 		sum += dists[k][1][0]
# 	if sum > 0:
# 		p_x.append(x_it)
# 		p_y.append(y_it)
# 	else:
# 		n_x.append(x_it)
# 		n_y.append(y_it)
# 	y_it += .02
# 	if y_it >= 1:
# 		y_it = -1
# 		x_it += .02

# blues, = plt.plot(p_x, p_y, 'bx')
# reds, = plt.plot(n_x, n_y, 'rx')
# ones, = plt.plot(pos_x, pos_y, 'bo', label='1')
# other, = plt.plot(neg_x, neg_y, 'ro', label='not 1')
# plt.show()

# ETEST
e_sum = 0
e_count = 0
for line in ftest:
	dists = []
	split = line.split(' ')
	# x_matrix.append([np.float32(split[1]), np.float32(split[2])])
	for j in range(0, len(x_matrix)):
		dist = np.power(np.power(np.float32(split[1]) - x_matrix[j][0], 2) + np.power(np.float32(split[2]) - x_matrix[j][1], 2), 0.5)
		dists.append([dist, y_matrix[j]])
	dists = sorted(dists, key=comparator)
	sum = 0
	for k in range(0, 7):
		sum += dists[k][1][0]
	if (split[0] == '1' and sum <= 0) or (split[0] == '0' and sum >= 0):
		e_sum += 1
	e_count += 1
print e_sum * 1.0/e_count