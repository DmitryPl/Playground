import numpy as np

from src.linopt.rosen import rosen2d, rosen2d_grad
from src.linopt.solvers import solver

x_a = np.array([0, 0])
x_b = np.array([-1, -1])
x_c = np.array([-0.9, -0.8])
x_d = np.array([-0.9, 1.1])

solver(x_a, rosen2d, rosen2d_grad, lambda x: 0.001)
