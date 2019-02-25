import numpy as np

a = np.ones([2, 3], np.float32)
a = [[0.1, 0.1, 0.1],
     [0.1, 0.2, 0.3]]

gamma = np.ones([1, 3], np.float32)
gamma = [[0.01, 0.02, 0.03]]

mean = np.ones([1, 3], np.float32)
mean = [[0.09, 0.08, 0.07]]

var = np.ones([1, 3], np.float32)
var = [[0.0001, 0.0003, 0.0004]]

z = np.subtract(a, mean)
s = np.sqrt(var)
normed = np.divide(z, s)
exp_normed = np.exp(np.multiply(-1, normed))
sigmoid_normed = np.divide(1, np.add(1, exp_normed))
item1 = np.subtract(1, sigmoid_normed)
item1 = np.multiply(item1, gamma)
item1 = np.multiply(item1, a)
item2 = np.multiply(sigmoid_normed, a)
y = np.add(item1, item2)

print y
