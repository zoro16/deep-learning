import numpy as np
import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)

started = time.time()
c = np.dot(a, b)
ended = time.time()

print("Vertorized version: {} ms".format(str(1000*(ended-started))))


c = 0

started = time.time()
for i in range(1000000):
    c += a[i] * b[i]
ended = time.time()

print("For loop version: {} ms".format(str(1000*(ended-started))))
