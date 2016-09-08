import numpy as np
import matplotlib.pyplot as plt

def f(t):
    return -2.0/3 * t - 1.0/3

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure(1)
plt.subplot(211)
plt.xlabel('x1')
plt.ylabel('x2')
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

plt.show()