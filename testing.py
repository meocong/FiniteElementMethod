import inspect
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import random

# def report(xk):
#     frame = inspect.currentframe().f_back
#     print(frame.f_locals['resid'])

N = 200
A = sparse.lil_matrix( (N, N) )
for _ in range(N):
    A[random.randint(0, N-1), random.randint(0, N-1)] = random.randint(1, 100)

b = np.random.randint(0, N-1, size = N)
print(b)
x, info = splinalg.cg(A, b)