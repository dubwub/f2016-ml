import math
import numpy as np
import matplotlib.pyplot as plt

neg_x = [1,0,0,-1]
neg_y = [0,1,-1,0]
pos_x = [0,0,-2]
pos_y = [2,-2,0]

# optional transform (6-1b):
for it in range(0, len(neg_x)):
	new_x = np.power(np.power(neg_x[it], 2) + np.power(neg_y[it], 2), 0.5)
	new_y = math.atan2(neg_y[it],neg_x[it])
	neg_x[it] = new_x
	neg_y[it] = new_y
for it in range(0, len(pos_x)):
	new_x = np.power(np.power(pos_x[it], 2) + np.power(pos_y[it], 2), 0.5)
	new_y = math.atan2(pos_y[it],pos_x[it])
	pos_x[it] = new_x
	pos_y[it] = new_y

dec_pos_x = []
dec_pos_y = []
dec_neg_x = []
dec_neg_y = []

x_it = -5
y_it = -5

# 1-NN
# while x_it < 3:
# 	min_dist = 10
# 	this_value = 0
# 	for it in range(0, len(neg_x)):
# 		dist = np.power(np.power(neg_x[it] - x_it, 2) + np.power(neg_y[it] - y_it, 2), .5)
# 		if dist < min_dist:
# 			min_dist = dist
# 			this_value = -1
# 	for it in range(0, len(pos_x)):
# 		dist = np.power(np.power(pos_x[it] - x_it, 2) + np.power(pos_y[it] - y_it, 2), .5)
# 		if dist < min_dist:
# 			min_dist = dist
# 			this_value = 1
# 	if this_value == 1:
# 		dec_pos_x.append(x_it)
# 		dec_pos_y.append(y_it)
# 	elif this_value == -1:
# 		dec_neg_x.append(x_it)
# 		dec_neg_y.append(y_it)
# 	y_it += 0.01
# 	if y_it > 3:
# 		y_it = -3
# 		x_it += 0.01

#3-NN

# comparing dists as opposed to secondary value
def comparator(pt1):
	return pt1[0]

while x_it < 5:
	dists = []
	for it in range(0, len(neg_x)):
		dists.append([np.power(np.power(neg_x[it] - x_it, 2) + np.power(neg_y[it] - y_it, 2), .5), -1])
	for it in range(0, len(pos_x)):
		dists.append([np.power(np.power(pos_x[it] - x_it, 2) + np.power(pos_y[it] - y_it, 2), .5), 1])
	this_value = 0
	dists = sorted(dists, key=comparator)
	for it in range(0, 1): # K for K-nn
		this_value += dists[it][1]
	if this_value > 0:
		dec_pos_x.append(x_it)
		dec_pos_y.append(y_it)
	elif this_value < 0:
		dec_neg_x.append(x_it)
		dec_neg_y.append(y_it)
	y_it += 0.01
	if y_it > 5:
		y_it = -5
		x_it += 0.01

plt.xlabel('x')
plt.ylabel('y')
dec_pos, = plt.plot(dec_pos_x, dec_pos_y, 'bx', label='')
dec_neg, = plt.plot(dec_neg_x, dec_neg_y, 'rx', label='')
pos_pts, = plt.plot(pos_x, pos_y, 'bo', label='+1')
neg_pts, = plt.plot(neg_x, neg_y, 'ro', label='-1')
plt.axis([-5, 5, -5, 5])
plt.legend(handles=[pos_pts, neg_pts])
plt.show()