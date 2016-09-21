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
numtests = 1000
for i in range(0, numtests):
	output = runExperiment()
	v0s.append(abs(output[0]/10.0 - 0.5))
	vrands.append(abs(output[1]/10.0 - 0.5))	
	vmins.append(abs(output[2]/10.0 - 0.5))
#print bins_
v0bins, xes = np.histogram(v0s, bins=bins_, density=True)
vrandbins, xes = np.histogram(vrands, bins=bins_, density=True)
vminbins, xes = np.histogram(vmins, bins=bins_, density=True)
v0bins = v0bins.tolist()
vrandbins = vrandbins.tolist()
vminbins = vminbins.tolist()
def divide(x): return x/10.0
v0bins = map(divide, v0bins)
vrandbins = map(divide, vrandbins)
vminbins = map(divide, vminbins)
print v0bins
v0bins.append(0)
vrandbins.append(0)
vminbins.append(0)
x = np.arange(0, 1, 0.01);
f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
#ax1.hist(np.array(v0s),bins=bins_, alpha=.3)
ax1.plot(x, 2*np.exp(-2*(x)**2 * 10))
ax1.plot(bins_, v0bins);
#ax2.hist(np.array(vrands), bins=bins_, alpha=.3)
ax2.plot(x, 2*np.exp(-2*(x)**2 * 10))
ax2.plot(bins_, vrandbins);
#ax3.hist(np.array(vmins), bins=bins_, alpha=.3)
ax3.plot(x, 2*np.exp(-2*(x)**2 * 10))
ax3.plot(bins_, vminbins);
f.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
ax1.set_title("v0, vrand, vmin Probability Approximations vs Hoeffding")
plt.xlabel("Epsilon")
plt.ylabel("Probability")
plt.show()


