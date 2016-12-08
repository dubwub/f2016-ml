import numpy as np

hidden = 2
x0 = [1, 1, 1]
w0 = []
s1 = []
w1 = []
x1 = [1]
s2 = []
x2 = 0

for i in range(0, 3):
	tmp = []
	for i in range(0, hidden):
		tmp.append(.25)
	w0.append(tmp)
for i in range(0, hidden+1):
	w1.append(.25)

w0 = np.matrix(w0)
w1 = np.matrix(w1)

gradients = [0 * w0, 0 * w1]

x0 = np.matrix([1, 1, 1])
x1 = [1]
s2 = []
x2 = 0
# forward prop
for j in range(0, hidden): # l = 0 to l = 1
	value = np.dot(x0, w0[:,j])
	x1.append(value)
x1 = np.matrix(x1)
print x1	
x2 = np.dot(x1, np.transpose(w1))
print x2
# e_in += np.power((x2 - y_matrix[i][0]), 2)
# x2 = x2.item(0)
# e_in = e_in.item(0)

d2 = 2*(x2 - 1)
print d2
d1 = d2 * w1[0,1:]
print d1
dew1 = np.dot(np.transpose(x0), np.matrix(d1))
print dew1
# gradients[0] += 1.0/len(x_matrix) * dew1
deW2 = np.dot(np.transpose(x1), np.matrix(d2))
# gradients[1] += 1.0/len(x_matrix) * np.transpose(deW2)
print deW2