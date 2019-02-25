#
import numpy as np

parallel_num = 2
M = 3
K = 2
batch_size = 2

a = np.ones([parallel_num, M, K], np.float32)
a = [[[0.1, 0.2],
      [0.01, 0.15],
      [1.1, 1.23]],
     [[1.0, 2.0],
      [1.1, 1.2],
      [0.1, 0.2]],
    ]

b = np.ones([parallel_num, M, K], np.float32)
b = [[[1.2, 2.4],
      [1.02, 3.25],
      [1.2, 1.25]],
     [[1.1, 2.4],
      [1.2, 2.2],
      [0.3, 1.2]],
    ]

print np.multiply(a[0], b[0])
print np.multiply(a[1], b[1])

