import numpy as np

# initialize random generator to get reproducible results
np.random.seed(42)

# generate matrix Q and vector b
n = 5
m = 2
A = np.random.rand(n, m)
b = 4 * np.random.rand(m)
Q = A.T @ A

# important parameters of the problem
eigs = np.linalg.eigvals(Q)
L = max(eigs)
mu = min(eigs)

print(f'b = {b}')
print(f'Q = \n{Q}')
print(f'mu = {mu}, L = {L}')

# minimum
x_star = -np.linalg.inv(Q) @ b
# some initial point
x_0 = x_star - [8, -1.5]
