import numpy as np
import matplotlib.pyplot as plt
import random
import time

# 6-16a

# num_points = 10000
# xes = []
# yes = []
# centers = [random.randint(0,num_points-1)]
# for i in range(0, num_points):
# 	xes.append(random.uniform(0,1))
# 	yes.append(random.uniform(0,1))

# 6-16b
num_points = 10000
xes = []
yes = []
centers = [random.randint(0,num_points-1)]
mean = 0
for i in range(0, num_points):
	if i % num_points/10 == 0:
		mean += 0.1 
	xes.append(np.random.normal(mean, 0.1, 1))
	yes.append(np.random.normal(mean, 0.1, 1))

start_time = time.time() # calculating starttime for clusters

# greedily get 10 clusters (already have one)
for i in range(0, 9):
	max_dist = 0
	max_index = -1
	for j in range(0, num_points):
		min_dist = 100
		for k in range(0, len(centers)):
			dist = np.power(np.power(xes[centers[k]] - xes[j], 2) + np.power(yes[centers[k]] - yes[j], 2), 0.5)
			if dist < min_dist:
				min_dist = dist # we want to maximize min distance to space out the clusters
		if min_dist > max_dist:
			max_dist = min_dist
			max_index = j
	centers.append(max_index)

for i in range(0, 9):
	print xes[centers[i]], yes[centers[i]]

# assign points to clusters (0-9)
clusters = []
cluster_xes = []
cluster_yes = []
cluster_radii = []
for i in range(0,10):
	clusters.append([])
	cluster_xes.append(0)
	cluster_yes.append(0)
	cluster_radii.append(0)

for i in range(0, num_points):
	min_dist = 100
	center = -1
	for k in range(0, len(centers)):
		dist = np.power(np.power(xes[centers[k]] - xes[i], 2) + np.power(yes[centers[k]] - yes[i], 2), 0.5)
		if dist < min_dist:
			min_dist = dist
			center = k
	clusters[center].append(i)
	if cluster_radii[center] < min_dist:
		cluster_radii[center] = min_dist
	cluster_xes[center] += xes[i]
	cluster_yes[center] += yes[i]

for i in range(0, 10):
	print len(clusters[i])
	if len(clusters[i]) == 0:
		continue
	cluster_xes[i] /= len(clusters[i])
	cluster_yes[i] /= len(clusters[i])

query_x = []
query_y = []
for i in range(0, num_points):
	query_x.append(random.uniform(0,1))
	query_y.append(random.uniform(0,1))

# BRUTE FORCE
# import time
# start_time = time.time()
# for i in range(0, num_points):
# 	if i % 100 == 0:
# 		print "Points completed:", i
# 	min_dist = 100
# 	min_index = -1
# 	for j in range(0, num_points):
# 		dist = np.power(np.power(xes[j] - query_x[i], 2) + np.power(yes[j] - query_y[i], 2), 0.5)
# 		if dist < min_dist:
# 			min_dist = dist
# 			min_index = j
# # 668.26699996 seconds
# print("--- %s seconds ---" % (time.time() - start_time))
# start_time = time.time()
# BRANCH AND BOUND
# import time
# start_time = time.time()
for i in range(0, num_points):
	if i % 100 == 0:
		print "Points completed:", i
	min_clust_dist = 100
	min_clust = -1
	for j in range(0, 10): # for each point, first find minimum cluster
		dist = np.power(np.power(xes[i] - cluster_xes[j], 2) + np.power(yes[i] - cluster_yes[j], 2), 0.5)
		if dist < min_clust_dist:
			min_clust_dist = dist
			min_clust = j
	min_dist = 100
	for j in range(0, len(clusters[min_clust])):
		dist = np.power(np.power(xes[i] - xes[clusters[min_clust][j]], 2) + np.power(yes[i] - yes[clusters[min_clust][j]], 2), 0.5)
		if dist < min_dist:
			min_dist = dist
	for j in range(0, 10):
		if j != min_clust:
			dist = np.power(np.power(xes[i] - cluster_xes[j], 2) + np.power(yes[i] - cluster_yes[j], 2), 0.5) - cluster_radii[j]
			if dist < min_dist:
				for k in range(0, len(clusters[j])):
					dist = np.power(np.power(xes[i] - xes[clusters[j][k]], 2) + np.power(yes[i] - yes[clusters[j][k]], 2), 0.5)
					if dist < min_dist:
						min_dist = dist
# 145.251000166 seconds
print("--- %s seconds ---" % (time.time() - start_time))