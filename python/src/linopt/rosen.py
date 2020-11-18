import numpy as np

array = np.ndarray


def rosen2d(x: array) -> array:
    return rosen2d_xy(x[0], x[1])


def rosen2d_xy(x: array, y: array) -> array:
    return 100 * (y - x ** 2) ** 2 + (1 - x) ** 2


def rosen2d_grad(x: array) -> array:
    return np.array([400.0 * (x[0] ** 2 - x[1]) + 2 * (x[0] - 1), 200 * (x[1] - x[0] ** 2)])


def rosen(x: array):
    """The Rosenbrock function"""
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0, axis=0)


def rosen_der(x: array) -> array:
    """The gradient of The Rosenbrock function"""
    xm = x[1: -1]
    xm_m1 = x[: - 2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1: -1] = 200 * (xm - xm_m1 ** 2) - 400 * (xm_p1 - xm ** 2) * xm - 2 * (1 - xm)
    der[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    der[-1] = 200 * (x[-1] - x[-2] ** 2)
    return der


def rosen_hess(x: array) -> array:
    """The hess matrix of The Rosenbrock function"""
    x = np.asarray(x)
    h = np.diag(-400 * x[:-1], 1) - np.diag(400 * x[:-1], -1)
    diagonal = np.zeros_like(x)
    diagonal[0] = 1200 * x[0] ** 2 - 400 * x[1] + 2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200 * x[1:-1] ** 2 - 400 * x[2:]
    h = h + np.diag(diagonal)
    return h
