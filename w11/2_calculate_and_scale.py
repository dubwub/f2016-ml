from decimal import Decimal
# f = open('training.txt', 'r')
# out = open('points.txt', 'w')
f = open('test.txt', 'r')
out = open('test_pts.txt', 'w')
ones_x = []
ones_y = []
other_x = []
other_y = []
# calculate symmetry about y-axis and x-axis
def getSymmetry(array):
	val = 0
	for i in range(0,8):
		for j in range(0, 16):
			if len(array) != 256:
				print len(array)
			val += abs(float(array[j*16 + i]) - float(array[(j+1)*16 - (i + 1)]))
	for i in range(0,16):
		for j in range(0,8):
			val += abs(float(array[j*16 + i]) - float(array[(15 - j)*16 - i]))
	val /= 128
	return val

def getIntensity(array):
	val = 0
	for i in range(0,256):
		# val += np.power(float(array[i]) + 1, 2)
		val += float(array[i]) + 1
	return val / 256

max_symmetry = 0
min_symmetry = -1
max_intensity = 0
min_intensity = -1
for line in f:
	line = line.split(' ')
	if line[0] == '1.0000':
		ones_x.append(getSymmetry(line[1:-1]))
		if ones_x[len(ones_x) - 1] > max_symmetry:
			max_symmetry = ones_x[len(ones_x) - 1]
		if ones_x[len(ones_x) - 1] < min_symmetry or min_symmetry == -1:
			min_symmetry = ones_x[len(ones_x) - 1]
		ones_y.append(getIntensity(line[1:-1]))
		if ones_y[len(ones_y) - 1] > max_intensity:
			max_intensity = ones_y[len(ones_y) - 1]
		if ones_y[len(ones_y) - 1] < min_intensity or min_intensity == -1:
			min_intensity = ones_y[len(ones_y) - 1]
	else:
		other_x.append(getSymmetry(line[1:-1]))
		if other_x[len(other_x) - 1] > max_symmetry:
			max_symmetry = other_x[len(other_x) - 1]
		if other_x[len(other_x) - 1] < min_symmetry or min_symmetry == -1:
			min_symmetry = other_x[len(other_x) - 1]
		other_y.append(getIntensity(line[1:-1]))
		if other_y[len(other_x) - 1] > max_intensity:
			max_intensity = other_y[len(other_y) - 1]
		if other_y[len(other_y) - 1] < min_intensity or min_intensity == -1:
			min_intensity = other_y[len(other_y) - 1]

for i in range(0, len(ones_x)):
	ones_x[i] -= min_symmetry
	ones_x[i] /= (max_symmetry - min_symmetry)
	ones_x[i] *= 2
	ones_x[i] -= 1
	ones_y[i] -= min_intensity
	ones_y[i] /= (max_intensity - min_intensity)
	ones_y[i] *= 2
	ones_y[i] -= 1
	out.write('1' + ' ' + str(ones_x[i]) + ' ' + str(ones_y[i]) + '\n')
	# x_matrix.append(legendreTransform(ones_x[i], ones_y[i]))
	# print legendreTransform(ones_x[i], ones_y[i])
	# y_matrix.append([1])
for i in range(0, len(other_x)):
	other_x[i] -= min_symmetry
	other_y[i] -= min_intensity
	other_x[i] /= (max_symmetry - min_symmetry)
	other_y[i] /= (max_intensity - min_intensity)
	other_x[i] *= 2
	other_y[i] *= 2
	other_x[i] -= 1
	other_y[i] -= 1
	out.write('0' + ' ' + str(other_x[i]) + ' ' + str(other_y[i]) + '\n')
	# x_matrix.append(legendreTransform(other_x[i], other_y[i]))
	# print legendreTransform(other_x[i], other_y[i])
	# y_matrix.append([-1])
