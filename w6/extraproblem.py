import numpy as np
import matplotlib.pyplot as plt
f = open('ZipDigits.train', 'r')

ones_x = []
ones_y = []
fives_x = []
fives_y = []

# calculate symmetry about y-axis and x-axis
def getSymmetry(array):
	val = 0
	for i in range(0,8):
		for j in range(0, 16):
			val += abs(float(array[j*16 + i]) - float(array[(j+1)*16 - (i + 1)]))
	for i in range(0,16):
		for j in range(0,8):
			val += abs(float(array[j*16 + i]) - float(array[(15 - j)*16 - i]))
	return val

def getIntensity(array):
	val = 0
	for i in range(0,256):
		# val += np.power(float(array[i]) + 1, 2)
		val += float(array[i]) + 1
	return val / 256

for line in f:
	line = line.split(' ')
	if line[0] == '1.0000':
		ones_x.append(getSymmetry(line[1:-1]))
		ones_y.append(getIntensity(line[1:-1]))
	elif line[0] == '5.0000':
		fives_x.append(getSymmetry(line[1:-1]))
		fives_y.append(getIntensity(line[1:-1]))

ones, = plt.plot(ones_x, ones_y, 'bo', label='1')
fives, = plt.plot(fives_x, fives_y, 'rx', label='5')
plt.xlabel('Symmetry')
plt.ylabel('Intensity')
plt.legend(handles=[ones, fives])
plt.show()