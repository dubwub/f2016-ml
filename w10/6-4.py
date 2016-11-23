import numpy as np
import matplotlib.pyplot as plt
import random

# details for semicircles
num_points = 500
rad = 10
thk = 5
sep = 5

# figure out how many points are going to be below line
neg_pt_x = []
neg_pt_y = []
pos_pt_x = []
pos_pt_y = []
neg_x = []
neg_y = []
pos_x = []
pos_y = []

def runExperiment(num_points, rad, thk, sep):
	neg_center = [rad+thk, rad+thk+sep]
	pos_center = [rad+2*thk, rad+thk]
	for i in range(0, num_points/2): # generate pts below line
		angle = random.uniform(0,np.pi)
		rad_pt = rad + random.uniform(0,thk)
		xval = neg_center[0] - rad_pt*np.cos(angle)
		yval = neg_center[1] + rad_pt*np.sin(angle)
		neg_pt_x.append(xval)
		neg_pt_y.append(yval)
		if angle < np.pi/32 or angle > np.pi * 31/32 or rad_pt - rad < thk/5:
			neg_x.append(xval)
			neg_y.append(yval)
	for i in range(num_points/2, num_points): # generate pts above line
		angle = random.uniform(0,np.pi)
		rad_pt = rad + random.uniform(0,thk)
		xval = pos_center[0] - rad_pt*np.cos(angle)
		yval = pos_center[1] - rad_pt*np.sin(angle)
		pos_pt_x.append(xval)
		pos_pt_y.append(yval)
		if angle < np.pi/32 or angle > np.pi * 31/32 or rad_pt - rad < thk/5:
			pos_x.append(xval)
			pos_y.append(yval)

runExperiment(2000,10,5,5)

dec_pos_x = []
dec_pos_y = []
dec_neg_x = []
dec_neg_y = []

x_it = 0
y_it = 0

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

while x_it < 40:
	if y_it < 15:
		dec_pos_x.append(x_it)
		dec_pos_y.append(y_it)
	elif y_it > 20:
		dec_neg_x.append(x_it)
		dec_neg_y.append(y_it)
	else:
		dists = []
		for it in range(0, len(neg_x)):
			dists.append([np.power(np.power(neg_x[it] - x_it, 2) + np.power(neg_y[it] - y_it, 2), .5), -1])
		for it in range(0, len(pos_x)):
			dists.append([np.power(np.power(pos_x[it] - x_it, 2) + np.power(pos_y[it] - y_it, 2), .5), 1])
		this_value = 0
		dists = sorted(dists, key=comparator)
		for it in range(0, 3): # K for K-nn
			this_value += dists[it][1]
		if this_value > 0:
			dec_pos_x.append(x_it)
			dec_pos_y.append(y_it)
		elif this_value < 0:
			dec_neg_x.append(x_it)
			dec_neg_y.append(y_it)
	y_it += 0.5
	if y_it > 40:
		y_it = 0
		x_it += 0.5

plt.xlabel('x')
plt.ylabel('y')
dec_pos, = plt.plot(dec_pos_x, dec_pos_y, 'bx', label='')
dec_neg, = plt.plot(dec_neg_x, dec_neg_y, 'rx', label='')
pos_pts, = plt.plot(pos_pt_x, pos_pt_y, 'bo', label='+1')
neg_pts, = plt.plot(neg_pt_x, neg_pt_y, 'ro', label='-1')
plt.axis([0, 40, 0, 40])
plt.legend(handles=[pos_pts, neg_pts])
plt.show()