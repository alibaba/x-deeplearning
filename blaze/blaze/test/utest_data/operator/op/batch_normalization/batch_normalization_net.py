import numpy as np

a = np.ones([2, 3], np.float32)
a = [[0.1, 0.1, 0.1],
     [0.1, 0.2, 0.3]]

gamma = np.ones([1, 3], np.float32)
gamma = [[0.01, 0.02, 0.03]]

beta = np.ones([1, 3], np.float32)
beta = [[0.03, 0.04, 0.05]]

mean = np.ones([1, 3], np.float32)
mean = [[0.09, 0.08, 0.07]]

var = np.ones([1, 3], np.float32)
var = [[0.0001, 0.0003, 0.0004]]

z = np.subtract(a, mean)
s = np.sqrt(np.add(var, 0.001))
normed = np.divide(z, s)

x = np.multiply(normed, gamma)
y = np.add(x, beta)

print y
