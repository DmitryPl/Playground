import math
from typing import Callable, List

import numpy as np

array = np.ndarray


def check_precision(precision: float, x_array: list, f: Callable) -> bool:
    return len(x_array) == 1 or abs(f(x_array[-1]) - f(x_array[-2])) > precision


def quad(q: array, b: array, x: array) -> array:
    return (0.5 * q.dot(x) + b).dot(x)


def quad_grad(q: array, b: array, x: array) -> array:
    return q.dot(x) + b


def quad_2d_xy(q: array, b: array, x: array, y: array) -> array:
    z = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i, j] = quad(q, b, np.array([x[i, j], y[i, j]]))
    return z


def rosen2d_xy(x: array, y: array) -> array:
    return 100 * (y - x ** 2) ** 2 + (1 - x) ** 2


def rosen(x: array):
    """The Rosenbrock function"""
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0, axis=0)


def rosen_grad(x: array) -> array:
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


def calc_gamma(q: array, b: array, fg: array, x: array, gamma: float) -> array:
    return quad(q, b, x + gamma * fg)


def gss(q: array, b: array, x: array, g_a: float, g_b: float, epsilon: float, f_grad: Callable) -> float:
    phi = 1 + (math.sqrt(5) - 1) / 2
    fg = -f_grad(x)

    while abs(g_a - g_b) >= epsilon:
        _a = g_b - (g_b - g_a) / phi
        _b = g_a + (g_b - g_a) / phi
        f1 = calc_gamma(q, b, fg, x, _a)
        f2 = calc_gamma(q, b, fg, x, _b)
        if f1 > f2:
            g_a = _a
        else:
            g_b = _b

    return (g_a + g_b) / 2


def armijo(x: array, alpha: float, theta: float, epsilon: float, f: Callable, f_grad: Callable) -> float:
    fg = f_grad(x)
    fx = f(x)
    _alpha = alpha
    while f(x - _alpha * fg) > fx - epsilon * _alpha * (np.dot(fg, fg)):
        _alpha *= theta
    return _alpha


def relaxation(q: array, b: array, x: array, f_grad: Callable) -> array:
    vec = -f_grad(x)
    return -(q.dot(x).dot(vec) + q.dot(vec).dot(x) + 2 * b.dot(vec)) / (2 * q.dot(vec).dot(vec))


def fib_list(n: int) -> array:
    fibonacci_numbers = [0, 1]
    for i in range(2, n):
        fibonacci_numbers.append(fibonacci_numbers[i - 1] + fibonacci_numbers[i - 2])
    return np.array(fibonacci_numbers)


def fibonacci(q: array, b: array, x: array, g_a: float, g_b: float, fib: array, fg: array) -> float:
    n = 10
    g1 = g_a + (g_b - g_a) * fib[n - 2] / fib[n]
    g2 = g_a + (g_b - g_a) * fib[n - 1] / fib[n]
    f1, f2 = calc_gamma(q, b, fg, x, g1), calc_gamma(q, b, fg, x, g2)

    while n > 1:
        n -= 1
        if f1 > f2:
            g_a = g1
            g1 = g2
            g2 = g_b - (g1 - g_a)
            f1, f2 = f2, calc_gamma(q, b, fg, x, g2)
        else:
            g_b = g2
            g2 = g1
            g1 = g_a + (g_b - g2)
            f2 = f1
            f1 = calc_gamma(q, b, fg, x, g1)

    return (g1 + g2) / 2


def werewolf(state: List[bool] = [True]) -> array:  # noqa
    state[0] = not state[0]
    return np.array([1, 0]) if state[0] else np.array([0, 1])


def changeable(x: array, q: array, b: array, state: List[bool]):
    vec = werewolf(state)
    return -(q.dot(x).dot(vec) + q.dot(vec).dot(x) + 2 * b.dot(vec)) / (2 * q.dot(vec).dot(vec))


def linear_convergence(f_k_residual):
    y = np.log(f_k_residual)
    x = np.arange(len(f_k_residual))
    x_mean, y_mean = np.mean(x), np.mean(y)
    xx_mean, xy_mean = np.mean(x.dot(x)), np.mean(x.dot(y))
    a = (xy_mean - x_mean * y_mean) / (xx_mean - x_mean * x_mean)
    b = y_mean - a * x_mean
    q = np.exp(a)
    c = np.exp(b)
    print(f"f(x_k) - f* = c * q^k = {c:.2f} * {q:.2f}^k")
    return x, y


def sublinear_convergence(f_k_residual, conv: Callable):
    y = f_k_residual[10:60]
    x = np.arange(len(f_k_residual[10:60]))
    x = conv(x)
    x_mean, y_mean = np.mean(x), np.mean(y)
    xx_mean, xy_mean = np.mean(x.dot(x)), np.mean(x.dot(y))
    a = (xy_mean - x_mean * y_mean) / (xx_mean - x_mean * x_mean)
    b = y_mean - a * x_mean
    print(f"f(x_k) - f* = a; b = {a:.2f}; {b:.2f}")
    return x, y
