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
print("running time in seconds:", time2)