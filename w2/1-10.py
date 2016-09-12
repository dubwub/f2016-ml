import math
import numpy as np
import matplotlib.pyplot as plt
import random

def flipCoin():
	heads = 0
	for i in range(0, 10):
		if random.uniform(0, 1) <= 0.5:
			heads += 1
	return heads

def runExperiment():
	#index 0 = v0, 1 = vrand, 2 = vmin
	results = []
	results.append(flipCoin())
	randIndex = random.randint(1, 999)
	minHeads = 10
	for i in range(0, 1000):
		output = flipCoin()
		if i == randIndex:
			results.append(output)
		if minHeads > output:
			minHeads = output
	results.append(minHeads)
	return results

# (a)
# results = runExperiment()
# print results[0], results[1], results[2]

# (b-e)
v0s = []
vrands = []
vmins = []
bins_ = []
for i in range(0, 10):
	bins_.append(i/10.0)
numtests = 10000
for i in range(0, numtests):
	output = runExperiment()
	v0s.append(output[0]/10.0)
	vrands.append(output[1]/10.0)	
	vmins.append(output[2]/10.0)
x = np.arange(0, 1, 0.01);
f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
ax1.hist(np.array(v0s),bins=bins_, alpha=.3)
# ax1.axvline(x=0.4, ymin=0, ymax = 1000, linewidth=2, color='k')
ax1.plot(x, 2*numtests*np.exp(-(x-.5)**2 * numtests))
ax2.hist(np.array(vrands), bins=bins_, alpha=.3)
ax2.plot(x, 2*numtests*np.exp(-(x-.5)**2 * numtests))
ax3.hist(np.array(vmins), bins=bins_, alpha=.3)
ax3.plot(x, 2*numtests*np.exp(-(x-.5)**2 * numtests))
f.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
ax1.set_title("v0, vrand, vmin Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()


