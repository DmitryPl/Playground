from typing import Callable, Dict

import numpy as np

from src.linopt.utils import armijo, check_precision, gss, relaxation, fibonacci, fib_list, werewolf, changeable

array = np.ndarray


def solver(
        x0: array,
        f: Callable,
        f_grad: Callable,
        gamma: Callable,
        iterations=70,
        precision=1e-6
) -> Dict[str, list]:
    """ Общая модель солверов на спуске """
    x = x0.copy()
    x_array, f_array = [x], [f(x)]
    k = 0

    while k < iterations and check_precision(precision, x_array, f):
        x = x + gamma(x) * (-f_grad(x))
        x_array.append(x)
        f_array.append(f(x))
        k += 1

    print(f'{k} iterations is taken, f(x^k) = {f_array[-1]}')
    return {'x_k': x_array, 'f_k': f_array}


def const_step_solver(x0: array, f: Callable, f_grad: Callable, iterations=70, precision=1e-6) -> Dict[str, list]:
    return solver(x0, f, f_grad, lambda x: 0.55, iterations, precision)


def armijo_solver(x0: array, f: Callable, f_grad: Callable, iterations=70, precision=1e-6):
    return solver(x0, f, f_grad, lambda x: armijo(x, 1., .5, .1, f, f_grad), iterations, precision)


def gss_solver(x0: array, f: Callable, f_grad: Callable, q: array, b: array, iterations=70, precision=1e-6):
    return solver(x0, f, f_grad, lambda x: gss(q, b, x, 0, 10., 1e-6, f_grad), iterations, precision)


def relaxation_solver(x0: array, f: Callable, f_grad: Callable, q: array, b: array, iterations=70, precision=1e-6):
    return solver(x0, f, f_grad, lambda x: relaxation(q, b, x, f_grad), iterations, precision)


def werewolf_solver(x0: array, f: Callable, q: array, b: array, iterations=70, precision=1e-6):
    fst, snd = [True], [True]
    return solver(x0, f, lambda x: -werewolf(fst), lambda x: changeable(x, q, b, snd), iterations, precision)


def fibonacci_solver(x0: array, f: Callable, q: array, b: array, iterations=70, precision=1e-6):
    fst, snd = [True], [True]
    fl = fib_list(15)
    gm = lambda x: fibonacci(q, b, x, -10., 10., fl, werewolf(snd))  # noqa
    return solver(x0, f, lambda x: -werewolf(fst), gm, iterations, precision)


def grad_solver(x0, f, f_grad, iterations=70, precision=1e-6):
    return solver(x0, f, f_grad, lambda x: armijo(x, 10., -.5, .01, f, f_grad), iterations, precision)


def coord_solver(x0, f, f_grad, iterations=70, precision=1e-6):
    x = x0.copy()
    x_array, f_array = [x], [f(x)]
    alpha, epsilon, theta = 10., 0.01, -.5
    f_star, tmp = 0, [True]
    k = 0

    while k < iterations:
        fg, gamma = werewolf(tmp), alpha
        while f(x + gamma * fg) > f(x) - epsilon * gamma * (np.dot(f_grad(x), fg)):
            gamma *= theta
        x = x + gamma * fg
        x_array.append(x)
        f_array.append(f(x))
        if abs(f(x_array[k]) - f_star) < precision:
            break
        k += 1

    print(f'{k} iterations is taken, f(x^k) = {f_array[-1]}')
    return {'x_k': x_array, 'f_k': f_array}
