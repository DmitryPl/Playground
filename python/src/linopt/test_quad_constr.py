import matplotlib.pyplot as plt
import numpy as np

from src.linopt.draw import draw_constr_line
from src.linopt.solvers import line_projection
from src.linopt.utils import quad, quad_grad

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
x_0 = x_star - [7, -1.5]

a = np.random.randn(2)
d = -2

data = line_projection(x_0, a, d, lambda x: quad(Q, b, x), lambda x: quad_grad(Q, b, x), precision=1e-15)
draw_constr_line(Q, b, a, d, x_0, x_star, data)
