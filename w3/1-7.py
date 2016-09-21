import numpy as np
import matplotlib.pyplot as plt
import random
import operator as op
def ncr(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom

bins = []
probabilities = []
for i in range(0, 6):
	bins.append(i/10.0)
	probabilities.append(0)

x = np.arange(0, .6, 0.1);

for i in range(0, 6):
	for j in range(0, 6):
		probability = .5**12 * (ncr(6,i) * ncr(6,j))
		print probability
		if abs(3-i) >= abs(3-j):
			for k in range(0, abs(3-i)+1):
				probabilities[k] += probability
		else:
			for k in range(0, abs(3-j)+1):
				probabilities[k] += probability

plt.xlabel('epsilon')
plt.ylabel('probability')
plt.plot(bins, probabilities)
plt.plot(x, 4*np.exp(-2*6*x**2) - 2*np.exp(-4*6*x**2))
plt.show()