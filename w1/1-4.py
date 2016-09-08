import numpy as np
import matplotlib.pyplot as plt
import random

# generate random line, we're going to plot from 0 to 100 in x,y [this is f]
num_points = 20 # change depending on problem
slope = random.uniform(-1, 1)
y_intercept = random.uniform(20, 80)
print slope
print y_intercept

# figure out how many points are going to be below line
num_points_neg = num_points / 2 + random.randint(-num_points/4, num_points/4) # at least 1/4 points in each sector
neg_x_pts = []
neg_y_pts = []
pos_x_pts = []
pos_y_pts = []

# not the best algorithm for generating random pts, this will probably run into some problems if the line goes out of bounds

for i in range(0, num_points_neg): # generate pts below line
	x = random.uniform(0, 100)
	y = x*slope + y_intercept
	random_y = 101
	while random_y >= 100:
		random_y = random.uniform(0, y)
	neg_x_pts.append(x)
	neg_y_pts.append(random_y)

for i in range(num_points_neg, num_points): # generate pts above line
	x = random.uniform(0, 100)
	y = x*slope + y_intercept
	random_y = random.uniform(y, 100)
	pos_x_pts.append(x)
	pos_y_pts.append(random_y)

plt.xlabel('x1')
plt.ylabel('x2')

t1 = np.arange(0.0, 100.0, 1)
f_line, = plt.plot(t1, t1*slope + y_intercept, 'k', label='f')
neg_pts, = plt.plot(neg_x_pts, neg_y_pts, 'ro', label='-1')
pos_pts, = plt.plot(pos_x_pts, pos_y_pts, 'bo', label='1')
plt.axis([0, 100, 0, 100])

def crossProduct(list1, list2):
	value = 0
	for i in range(0, len(list1)):
		value += list1[i] * list2[i]
	return value

# begin PLA (we're determining g)
weights = [0, 0, 0]
iterations = 0
reiterate = False
while True:
	# if iterations > 10000:
	# 	break
	# if weights[2] != 0:
		# g_slope = -1.0*weights[1]/weights[2]
		# g_intercept = -1.0*weights[0]/weights[2]
		# print g_slope, g_intercept
	reiterate = False # tells loop whether to break early and continue
	for i in range(0, len(neg_x_pts)):
		pt = [1, neg_x_pts[i], neg_y_pts[i]]
		if crossProduct(weights, pt) >= 0:
			# print 'fail1'
			# print crossProduct(weights, pt), 1
			# print pt
			for it in range(0, 3):
				weights[it] += -1.0 * pt[it]
			reiterate = True
			break
	if reiterate == True:
		iterations += 1
		continue
	# print 'done here'
	for i in range(0, len(pos_x_pts)):
		pt = [1, pos_x_pts[i], pos_y_pts[i]]
		if crossProduct(weights, pt) <= 0:
			# print 'fail2'
			# print crossProduct(weights, pt)
			for it in range(0, 3):
				weights[it] += 1.0 * pt[it]
			reiterate = True
			break
	if reiterate == True:
		iterations += 1
		continue
	else:
		break
print iterations
g_slope = -1.0*weights[1]/weights[2]
g_intercept = -1.0*weights[0]/weights[2]
print g_slope, g_intercept
g_line, = plt.plot(t1, t1*g_slope + g_intercept, 'g', label='g')
# axes = plt.gca()
# axes.set_xlim([-100,100])
# axes.set_ylim([-100,100])
plt.legend(handles=[f_line, neg_pts, pos_pts, g_line])
plt.show()