import numpy as np
import matplotlib.pyplot as plt
f = open('points.txt', 'r')

x_matrix = []
y_matrix = []

for line in f:
	split = line.split(' ')
	x_matrix.append([np.float32(split[1]), np.float32(split[2])])
	if split[0] == '1':
		y_matrix.append([1])
	else:
		y_matrix.append([-1])

# pt1 is [distance, y_val]
def comparator(pt1):
	return pt1[0]

distance_matrix = []
for i in range(0, len(x_matrix)):
	dists = []
	for j in range(0, len(x_matrix)):
		if i == j:
			continue
		dist = np.power(np.power(x_matrix[i][0] - x_matrix[j][0], 2) + np.power(x_matrix[i][1] - x_matrix[j][1], 2), 0.5)
		dists.append([dist, y_matrix[j]])
	dists = sorted(dists, key=comparator)
	# print dists
	distance_matrix.append(dists)

# print distance_matrix

# now that we have all the sorted distances from each point, we can find the optimal k-NN
min_error = 1
min_k = 0
k = 1
while k < len(x_matrix)/2:
	error = 0
	for i in range(0, len(y_matrix)): # distance matrix is 300 x 299
		sum = 0
		for j in range(0, k):
			# print distance_matrix[i][j][1][0]
			sum += distance_matrix[i][j][1][0]
		# print sum, y_matrix[i][0]
		if (sum < 0 and y_matrix[i][0] > 0) or (sum > 0 and y_matrix[i][0] < 0):
			error += 1.0/len(y_matrix)
	print error
	if error < min_error:
		min_error = error
		min_k = k
	k += 2