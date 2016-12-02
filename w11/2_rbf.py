import random
import numpy as np
import matplotlib.pyplot as plt
f = open('points.txt', 'r')
ftest = open('test_pts.txt', 'r')

x_matrix = []
y_matrix = []
kmeans_array = []

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

# ecv_array = []
# for i in range(0, len(x_matrix)):
# 	tmp = x_matrix[:]
# 	del tmp[i]
# 	ecv_array.append(np.array(tmp))
ecv_array = np.array(x_matrix)

# ones, = plt.plot(pos_x, pos_y, 'bo', label='1')
# other, = plt.plot(neg_x, neg_y, 'rx', label='not 1')
# plt.show()
# END UNCOMMENT BLOCK

# K-MEAN CLUSTER ALGORITHM FROM ONLINE
def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters
 
def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu
 
def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))
 
def find_centers(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)
# END FROM ONLINE

# centers = find_centers(kmeans_array, 3)[0]
# print centers

# pt1 is [distance, y_val]
# def comparator(pt1):
# 	return pt1[0]

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

np_y = np.matrix(y_matrix)

# FIND MIN E_CV
min_error = -1
min_centers = []
max_error = 0
min_k = 0
errors = []
ks = []
k = 1
while k < 80:
	centers = find_centers(ecv_array, k)[0] # run kmeans on data set minus this point
	RBF_matrix = []
	for i in range(0, len(x_matrix)):
		data = [1]
		for j in range(0, k):
			dist = np.power(np.power(x_matrix[i][0] - centers[j][0], 2) + np.power(x_matrix[i][1] - centers[j][1], 2), 0.5)
			dist /= (2/np.power(k,.5))
			gaussian = np.exp([-.5 * np.power(dist, 2)])[0]
			data.append(gaussian)
		RBF_matrix.append(data)
	RBF_matrix = np.matrix(RBF_matrix)
	# w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(RBF_matrix), RBF_matrix)), np.transpose(RBF_matrix)), np_y)
	# guess = np.dot(w, np_y)
	# print np.dot(np.transpose(RBF_matrix), np.linalg.inv(np.dot(RBF_matrix, np.transpose(RBF_matrix)))).shape
	# print RBF_matrix.shape
	w = np.dot(np.transpose(np.dot(np.transpose(RBF_matrix), np.linalg.inv(np.dot(RBF_matrix, np.transpose(RBF_matrix))))), np.transpose(RBF_matrix))
	# print w.shape
	guess = np.dot(w, np_y)
	real_guess = []
	for i in range(0, 300):
		if guess[i] >= 0:
			real_guess.append([1])
		else:
			real_guess.append([-1])
	guess = np.matrix(real_guess)

	ecv = 0
	for i in range(0, 300):
		# add_to_ecv = np.sum(y_matrix[i]) - np.sum(guess[i])
		add_to_ecv = 0
		if np.sum(y_matrix[i]) != np.sum(guess[i]):
			add_to_ecv = 1
		else:
			add_to_ecv = 0
		# print y_matrix[i], guess[i]
		# print add_to_ecv
		add_to_ecv = np.sum(add_to_ecv)
		add_to_ecv /= (1 - np.sum(w.item((i,i))))
		add_to_ecv = np.power(add_to_ecv, 2)
		# print add_to_ecv
		ecv += add_to_ecv
	ecv /= 300
	errors.append(ecv)
	ks.append(k)
	if ecv > max_error:
		max_error = ecv
	if ecv < min_error or min_error == -1:
		min_centers = centers
		min_error = ecv
		min_k = k
	print "k:", k, "min_error:", min_error, "min_k:", min_k
	k += 1

for i in range(0, len(errors)):
	errors[i] /= max_error

print min_error, min_k

# UNCOMMENT BELOW FOR ECV GRAPH
kgraph, = plt.plot(ks, errors, 'bo', label='k')
plt.legend(handles=[kgraph])
plt.axis([0, 80, 0, .1])
plt.xlabel('k (RBF)')
plt.ylabel('Ecv')
plt.show()
# END UNCOMMENT BLOCK

# RUNNING JUST ON k = 12, TRYING TO FIND E_IN
error = 0
k = min_k
centers = min_centers
# centers = find_centers(ecv_array, k)[0] # run kmeans on data set minus this point
RBF_matrix = []
for i in range(0, len(x_matrix)):
	data = [1]
	for j in range(0, k):
		dist = np.power(np.power(x_matrix[i][0] - centers[j][0], 2) + np.power(x_matrix[i][1] - centers[j][1], 2), 0.5)
		dist /= (2/np.power(k,.5))
		gaussian = np.exp([-.5 * np.power(dist, 2)])[0]
		data.append(gaussian)
	RBF_matrix.append(data)
RBF_matrix = np.matrix(RBF_matrix)
w = np.dot(np.transpose(np.dot(np.transpose(RBF_matrix), np.linalg.inv(np.dot(RBF_matrix, np.transpose(RBF_matrix))))), np.transpose(RBF_matrix))
# print np.dot(np.linalg.inv(np.dot(RBF_matrix, np.transpose(RBF_matrix))), RBF_matrix).shape
wreg = np.dot(np.transpose(np.dot(np.linalg.inv(np.dot(RBF_matrix, np.transpose(RBF_matrix))), RBF_matrix)), np_y)
# print wreg
guess = np.dot(w, np_y)
# print w.shape
real_guess = []
for i in range(0, 300):
	if guess[i] >= 0:
		real_guess.append([1])
	else:
		real_guess.append([-1])
guess = np.matrix(real_guess)

# error = 0.0
# for i in range(0, 300):
# 	print np.sum(y_matrix[i]), np.sum(guess[i])
# 	if (np.sum(y_matrix[i]) > 0 and np.sum(guess[i]) < 0) or (np.sum(y_matrix[i]) < 0 and np.sum(guess[i]) > 0):
# 		error += 1.0
# error /= 300
# print error, k

p_x = []
p_y = []
n_x = []
n_y = []

# # SHOWING DECISION BOUNDS
x_it = -1
y_it = -1
while x_it <= 1:
	data = [1]
	for j in range(0, k):
		dist = np.power(np.power(x_it - centers[j][0], 2) + np.power(y_it - centers[j][1], 2), 0.5)
		dist /= (2/np.power(k,.5))
		gaussian = np.exp([-.5 * np.power(dist, 2)])[0]
		data.append(gaussian)
	# print wreg, data
	sum = np.sum(np.dot(np.transpose(wreg), np.transpose(np.matrix(data))))
	# print sum
	if sum > 0:
		p_x.append(x_it)
		p_y.append(y_it)
	else:
		n_x.append(x_it)
		n_y.append(y_it)
	y_it += .02
	if y_it >= 1:
		y_it = -1
		x_it += .02

blues, = plt.plot(p_x, p_y, 'bx')
reds, = plt.plot(n_x, n_y, 'rx')
ones, = plt.plot(pos_x, pos_y, 'bo', label='1')
other, = plt.plot(neg_x, neg_y, 'ro', label='not 1')
plt.show()

# ETEST
# e_sum = 0
# e_count = 0
# for line in ftest:
# 	dists = []
# 	split = line.split(' ')
# 	# x_matrix.append([np.float32(split[1]), np.float32(split[2])])
# 	for j in range(0, len(x_matrix)):
# 		dist = np.power(np.power(np.float32(split[1]) - x_matrix[j][0], 2) + np.power(np.float32(split[2]) - x_matrix[j][1], 2), 0.5)
# 		dists.append([dist, y_matrix[j]])
# 	dists = sorted(dists, key=comparator)
# 	sum = 0
# 	for k in range(0, 7):
# 		sum += dists[k][1][0]
# 	if (split[0] == '1' and sum <= 0) or (split[0] == '0' and sum >= 0):
# 		e_sum += 1
# 	e_count += 1
# print e_sum * 1.0/e_count