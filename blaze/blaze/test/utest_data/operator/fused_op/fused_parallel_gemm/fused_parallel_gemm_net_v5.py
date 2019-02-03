#
import numpy as np

parallel_num = 3
M = 3
K = 2
N = 3

a = np.ones([M, K], np.float32)
a = [[0.1, 0.2],
      [0.01, 0.15],
      [1.1, 1.23]]

b = np.ones([parallel_num, K, N], np.float32)
b = [[[1.2, 0.2, 1.15],
      [0.01, 0.1, 2.1]],
     [[0.1, 0.01, 0.7],
      [1.1, 2.1, -1.5]],
     [[1.1, -2.1, -7],
      [2.1, 1.1, -0.8]]]

c = np.matmul(a, b)
print c
