from typing import Callable

import numpy as np

from src.linopt.utils import rosen, rosen_grad

array = np.ndarray


def checker(x: array, f: Callable, dx: float) -> array:
    return np.array([(f(x + np.array([dx, 0])) - f(x)) / dx, (f(x + np.array([0, dx])) - f(x)) / dx])


x_a = np.array([0, 0])
x_b = np.array([-1, -1])
x_c = np.array([-0.9, -0.8])
x_d = np.array([-0.9, 1.1])

for point in [x_a, x_b, x_c, x_d]:
    first = checker(point, rosen, 1e-10)
    second = rosen_grad(point)

    print(f'checker: {first[0]:.2f}, {first[1]:.2f}, '
          f'rosen: {second[0]:.2f}, {second[1]:.2f}\n')
