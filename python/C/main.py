#!/usr/bin/env python3

from ctypes import c_void_p, c_double, c_int, cdll
from numpy.ctypeslib import ndpointer
import numpy as np
import time


def py_sum(matrix: np.array, n: int, m: int) -> np.array:
    result = np.zeros(n)
    for i in range(0, n):
        for j in range(0, m):
            result[i] += matrix[i][j]
    return result


n = 10000
m = 10000
matrix = np.random.randn(n, m)

time1 = time.time()
py_result = py_sum(matrix, n, m)
time2 = time.time() - time1
print("py running time in seconds:", time2)
py_time = time2

lib = cdll.LoadLibrary("src/c_sum.so")
c_sum = lib.c_sum
c_sum.restype = ndpointer(dtype=c_double, shape=(n, ))

time1 = time.time()
result = c_sum(c_void_p(matrix.ctypes.data), c_int(n), c_int(m))
time2 = time.time() - time1
print("c  running time in seconds:", time2)

c_time = time2
print("speedup:", py_time / c_time)
