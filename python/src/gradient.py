import math
from typing import Callable, Dict, Tuple

import numpy as np

inv_phi = (math.sqrt(5) - 1) / 2  # 1 / phi
inv_phi2 = (3 - math.sqrt(5)) / 2  # 1 / phi^2


def quad(q: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    return (0.5 * q.dot(x) + b).dot(x)


def quad_grad(q: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    return q.dot(x) + b


def const_step_solver(x0: np.ndarray, f: Callable, f_grad: Callable, iterations=70) -> Dict[str, list]:
    """
    Solver example.
    Input:
        x0 - initial point
        f - function with signature f(x)
        f_grad - function with signature f_grad(x)
        params - reserved to parametrize this function
    Output:
        dictionary with keys:
            'x_k': list of intermediate points
            'f_k': list of intermediate function values
    """
    x = x0.copy()
    x_array, f_array = [x], [f(x)]
    gamma, k = 0.55, 0

    while k < iterations:
        x = x + gamma * (-f_grad(x))
        x_array.append(x)
        f_array.append(f(x))
        k += 1

    print(f'{k} iterations is taken, f(x^k) = {f_array[-1]}')
    return {'x_k': x_array, 'f_k': f_array}


def gss(f: Callable, a: float, b: float, tol=1e-5) -> Tuple[float, float]:
    """
    Given a function f with a single local minimum in
    the interval [a,b], gss returns a subset interval
    [c,d] that contains the minimum with d-c <= tol.
    """
    (a, b) = (min(a, b), max(a, b))
    h = b - a
    if h <= tol:
        return a, b

    # Required steps to achieve tolerance
    n = int(math.ceil(math.log(tol / h) / math.log(inv_phi)))

    c = a + inv_phi2 * h
    d = a + inv_phi * h
    yc = f(c)
    yd = f(d)

    for k in range(n - 1):
        if yc < yd:
            b = d
            d = c
            yd = yc
            h = inv_phi * h
            c = a + inv_phi2 * h
            yc = f(c)
        else:
            a = c
            c = d
            yc = yd
            h = inv_phi * h
            d = a + inv_phi * h
            yd = f(d)

    if yc < yd:
        return a, d
    else:
        return c, b
