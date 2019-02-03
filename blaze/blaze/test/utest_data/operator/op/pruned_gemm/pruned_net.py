import numpy as np

## a: 1 * 2
## b: 2 * 3
## w1: 2 * 2
## w2: 3 * 2

## bias: 1 * 2

a = np.ones([1, 2], np.float32)
a = [[0.1, 0.2]]

b = np.ones([2, 3], np.float32)
b = [[0.2, 0.01, 0.03],
     [-1.0, -0.01, -0.21]]

w1 = np.ones([2, 2], np.float32)
w1 = [[0.1, 0.2],
     [-0.3, 0.4]]

w2 = np.ones([3, 2], np.float32)
w2 = [[0.2, 0.3],
      [-0.1, 0.2],
      [-0.11, 0.12]]

bias = np.ones([1, 2], np.float32)
bias = [[0.001, 0.002]]

o1 = np.matmul(a, w1)
o2 = np.matmul(b, w2)
output = np.add(o1, o2)
output = np.add(output, bias)
print output

