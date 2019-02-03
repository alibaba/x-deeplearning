#
import numpy as np

parallel_num = 2
M = 3
K = 2
N = 2
batch_size = 2

a = np.ones([parallel_num, 1, M, K], np.float32)
a = [[[[0.1, 0.2],
       [0.01, 0.15],
       [1.1, 1.23]]],
     [[[1.0, 2.0],
       [1.1, 1.2],
       [0.1, 0.2]]],
    ]

b = np.ones([parallel_num, N, M, 1], np.float32)
b = [[[[0.4],
       [0.5],
       [0.6]],
      [[0.7],
       [0.8],
       [0.9]]],
     [[[0.1],
       [0.2],
       [0.3]], 
      [[1.0],
       [1.1],
       [1.2]]],
    ]

s = np.multiply(a, b)

print np.sum(s, axis=2)
