#
import numpy as np

parallel_num = 2
M = 3
K = 2
N = 1
batch_size = 2

a = np.ones([parallel_num, M, K], np.float32)
a = [[[0.1, 0.2],
      [0.01, 0.15],
      [1.1, 1.23]],
     [[1.0, 2.0],
      [1.1, 1.2],
      [0.1, 0.2]],
    ]

b = np.ones([parallel_num, batch_size, K, N], np.float32)
b = [[[[1.0],
       [2.0]],
      [[0.1],
       [1.1]]],
     [[[1.2],
       [2.1]],
      [[1.1],
       [0.1]]]]

print np.matmul(a[0], b[0])
print np.matmul(a[1], b[1])

