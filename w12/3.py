import random
import numpy as np
import matplotlib.pyplot as plt

p_x = []
p_y = []
n_x = []
n_y = []

x_it = -1
y_it = -1
while x_it <= 1:
	if x_it < .01 and x_it > -.01:
		p_x.append(x_it)
		p_y.append(y_it)
	if np.power(x_it,3) - y_it < .01 and np.power(x_it,3) - y_it > -.001:
		n_x.append(x_it)
		n_y.append(y_it)
	y_it += .02
	if y_it >= 1:
		y_it = -1
		x_it += .02

blues, = plt.plot(p_x, p_y, 'bx')
reds, = plt.plot(n_x, n_y, 'rx')
plt.plot([-1, 1], [0, 0], 'bo')
plt.show()