import numpy as np

from src.linopt.draw import draw_rosen
from src.linopt.solvers import coord_solver, grad_solver
from src.linopt.utils import rosen, rosen_grad

x_a = np.array([0, 0])
x_b = np.array([-1, -1])
x_c = np.array([-0.9, -0.8])
x_d = np.array([-0.9, 1.1])

# zeros = solver(x_a, rosen, rosen_der, lambda x: 1e-4, 1e5, 1e-10)
first = grad_solver(x_a, rosen, rosen_grad, 1e5, 1e-10)
second = coord_solver(x_a, rosen, rosen_grad, 1e5, 1e-10)

draw_rosen(first, second)
