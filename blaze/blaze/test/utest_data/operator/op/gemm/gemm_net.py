import numpy as np

a = np.ones([2, 3], np.float32)
a = [[0.1, 0.2, -0.3],
     [0.4, 0.5, 0.6]]

b = np.ones([3, 2], np.float32)
b = [[0.1, 0.2],
     [-0.3, 0.4],
     [0.5, -0.6]]

bias = np.ones([1, 2], np.float32)
bias = [[0.001, 0.002]]

c = np.matmul(a, b)
output = np.add(c, bias)
print output

