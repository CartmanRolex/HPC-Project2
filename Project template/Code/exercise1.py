# Exercise I: Matrices in Python
import numpy as np
from time import perf_counter as tic
# We try different ways of assigning the matrix (these are some of the options)

size = (2,4)

# Manually entering the elements
t0 = tic()
M0 = np.empty(size)
c = 1
for i in range(size[0]):
    for j in range(size[1]):
        M0[i, j] = c
        c += 1
t1 = tic() - t0
print("Time taken entering manually each element: " + f"{t1:.2e}" + " s.")

# Filling the matrix row wise
t0 = tic()
M1 = np.empty(size)
for i in range(size[0]):
    M1[i, :] = np.arange(i*size[1] + 1, (i+1)*size[1] + 1)
t1 = tic() - t0
print("Time taken filling the matrix row wise: " + f"{t1:.2e}" + " s.")

# Filling the matrix column wise
t0 = tic()
M2 = np.empty(size)
for j in range(size[1]):
    M2[:, j] = np.arange( j+1, (size[0]-1)*size[1] + j + 2, size[1] )
t1 = tic() - t0
print("Time taken filling the matrix row wise: " + f"{t1:.2e}" + " s.")


# Using np arrange and np resize
t0 = tic()
M1 = np.resize(np.arange(1, 9), (4,2))
t1 = tic() - t0
print("Time taken using np.arange and np.resize: " + f"{t1:.2e}" + " s.")

# Using linspace
t0 = tic()
M2 = np.linspace( (1, 5), (4, 8), 4 ).T
t1 = tic() - t0
print("Time taken using np.linspace: " + f"{t1:.2e}" + " s.")
