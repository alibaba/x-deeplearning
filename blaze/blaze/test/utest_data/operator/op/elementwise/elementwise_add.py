import numpy as np

a = np.ones([2, 3, 2], np.float32)
a = [[[0.1, 0.2],
      [-0.3, 0.4],
      [0.5, 0.6]],
     [[0.7, 0.8],
      [0.9, 1.0],
      [1.1, 1.2]]]

b = np.ones([2], np.float32)
b = [0.2, 0.3]

print np.add(a, b)
