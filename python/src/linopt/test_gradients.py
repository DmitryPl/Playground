import numpy as np

from src.linopt.draw import draw_gradients
from src.linopt.solvers import const_step_solver, armijo_solver, gss_solver, relaxation_solver, werewolf_solver, \
    fibonacci_solver
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
x_0 = x_star - [8, -1.5]

# first = const_step_solver(x_0, lambda x: quad(Q, b, x), lambda x: quad_grad(Q, b, x))
# second = armijo_solver(x_0, lambda x: quad(Q, b, x), lambda x: quad_grad(Q, b, x))

# first = gss_solver(x_0, lambda x: quad(Q, b, x), lambda x: quad_grad(Q, b, x), Q, b)
# second = relaxation_solver(x_0, lambda x: quad(Q, b, x), lambda x: quad_grad(Q, b, x), Q, b)

first = fibonacci_solver(x_0, lambda x: quad(Q, b, x), Q, b)
second = werewolf_solver(x_0, lambda x: quad(Q, b, x), Q, b)

draw_gradients(Q, b, x_0, x_star, first, second)
