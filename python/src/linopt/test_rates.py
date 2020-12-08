import numpy as np

from src.linopt.draw import draw_lines
from src.linopt.solvers import const_step_solver, armijo_solver, gss_solver, relaxation_solver, fibonacci_solver, \
    werewolf_solver
from src.linopt.utils import quad, quad_grad, linear_convergence

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

first = const_step_solver(x_0, lambda x: quad(Q, b, x), lambda x: quad_grad(Q, b, x))
second = armijo_solver(x_0, lambda x: quad(Q, b, x), lambda x: quad_grad(Q, b, x))
residual1 = ('const step', first['f_k'] - quad(Q, b, x_star))
residual2 = ('armijo', second['f_k'] - quad(Q, b, x_star))
linear_convergence(residual1[1])
linear_convergence(residual2[1])

first = gss_solver(x_0, lambda x: quad(Q, b, x), lambda x: quad_grad(Q, b, x), Q, b)
second = relaxation_solver(x_0, lambda x: quad(Q, b, x), lambda x: quad_grad(Q, b, x), Q, b)
residual3 = ('gss', first['f_k'] - quad(Q, b, x_star))
residual4 = ('relaxation', second['f_k'] - quad(Q, b, x_star))
linear_convergence(residual3[1])
linear_convergence(residual4[1])

draw_lines([residual1, residual2, residual4])

first = fibonacci_solver(x_0, lambda x: quad(Q, b, x), Q, b)
second = werewolf_solver(x_0, lambda x: quad(Q, b, x), Q, b)
linear_convergence(first['f_k'] - quad(Q, b, x_star))
linear_convergence(second['f_k'] - quad(Q, b, x_star))
