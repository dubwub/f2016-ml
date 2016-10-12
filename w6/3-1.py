import numpy as np
import matplotlib.pyplot as plt
import random

# details for semicircles
num_points = 2000
rad = 10
thk = 5
sep = 5

# figure out how many points are going to be below line
neg_x_pts = []
neg_y_pts = []
pos_x_pts = []
pos_y_pts = []
pts_x = []
pts_y = []

def runExperiment(num_points, rad, thk, sep):
	neg_center = [rad+thk, rad+thk+sep]
	pos_center = [rad+2*thk, rad+thk]
	for i in range(0, num_points/2): # generate pts below line
		angle = random.uniform(0,np.pi)
		rad_pt = rad + random.uniform(0,thk)
		xval = neg_center[0] - rad_pt*np.cos(angle)
		yval = neg_center[1] + rad_pt*np.sin(angle)
		pts_x.append([1, xval, yval])
		neg_x_pts.append(xval)
		neg_y_pts.append(yval)
		pts_y.append([-1.0])
	for i in range(num_points/2, num_points): # generate pts above line
		angle = random.uniform(0,np.pi)
		rad_pt = rad + random.uniform(0,thk)
		xval = pos_center[0] - rad_pt*np.cos(angle)
		yval = pos_center[1] - rad_pt*np.sin(angle)
		pts_x.append([1, xval, yval])
		pos_x_pts.append(xval)
		pos_y_pts.append(yval)
		pts_y.append([1.0])

runExperiment(2000,10,5,5)
plt.xlabel('x1')
plt.ylabel('x2')

t1 = np.arange(-10, 50, 1)
neg_pts, = plt.plot(neg_x_pts, neg_y_pts, 'ro', label='-1')
pos_pts, = plt.plot(pos_x_pts, pos_y_pts, 'bo', label='1')
plt.axis([-10, 50, -10, 50])

# def crossProduct(list1, list2):
# 	value = 0
# 	for i in range(0, len(list1)):
# 		value += list1[i] * list2[i]
# 	return value

# begin PLA (3.1a)
# weights = [0, 0, 1]
# iterations = 0
# reiterate = False
# while True:
# 	if iterations > 3000:
# 		break
# 	# if weights[2] != 0:
# 		# g_slope = -1.0*weights[1]/weights[2]
# 		# g_intercept = -1.0*weights[0]/weights[2]
# 		# print g_slope, g_intercept
# 	reiterate = False # tells loop whether to break early and continue
# 	for i in range(0, len(pts_x)):
# 		pt = [1.0, 1.0*pts_x[i][0], 1.0*pts_x[i][1]]
# 		if (crossProduct(weights, pt) >= 0 and pts_y[i] == -1) or (crossProduct(weights, pt) <= 0 and pts_y[i] == 1):
# 			for it in range(0, 3):
# 				weights[it] += pts_y[i] * pt[it]
# 			reiterate = True
# 			break
# 	if reiterate == True:
# 		iterations += 1
# 		continue
# 	else:
# 		break
# print iterations
# g_slope = -1.0*weights[1]/weights[2]
# g_intercept = -1.0*weights[0]/weights[2]

# begin linear regression (3.1b)
x_matrix = np.matrix(pts_x)
y_matrix = np.matrix(pts_y)
weights = (np.linalg.inv(np.transpose(x_matrix) * x_matrix) * np.transpose(x_matrix)) * y_matrix
print weights
g_slope = -1.0*weights[1][0]/weights[2][0]
g_slope = g_slope.item(0, 0)
g_intercept = -1.0*weights[0][0]/weights[2][0]
g_intercept = g_intercept.item(0,0)

print g_slope, g_intercept
g_line, = plt.plot(t1, t1*g_slope + g_intercept, 'g', label='g')

plt.legend(handles=[neg_pts, pos_pts, g_line])
plt.show()