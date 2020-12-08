from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np

from src.linopt.utils import quad_2d_xy, quad, quad_grad, rosen2d_xy


def draw_gradients(q, b, x_0, x_star, first, second):
    # prepare image
    _, axes = plt.subplots(1, 2, figsize=(15, 5))
    # plot contours
    X, Y = np.meshgrid(np.linspace(-5, 15, 400), np.linspace(-10, 3, 100))  # noqa
    Z_quad = quad_2d_xy(q, b, X, Y)  # noqa

    # some step-size
    gamma = 0.6
    # levels for contour lines
    levels = quad(q, b, x_star) + np.linspace(0, 10, 6) ** 2

    # on both sub-plots plot:
    for k in (0, 1):
        # having equal axes is important to see if lines orthogonal or not
        axes[k].axis('equal')
        # minimum
        axes[k].plot(x_star[0], x_star[1], '*')
        # initial point
        axes[k].plot(x_0[0], x_0[1], 'or')
        # a segment in anti-gradient direction
        axes[k].plot((x_0[0], (x_0 - gamma * quad_grad(q, b, x_0))[0]),
                     (x_0[1], (x_0 - gamma * quad_grad(q, b, x_0))[1]), 'r')
        # plot contour lines
        axes[k].contour(X, Y, Z_quad, levels)

    x_data = []
    for j, data in enumerate([first, second]):
        # convert list of vectors to 2D-array of dimension N x 2
        _x_data = np.array(data['x_k'])
        # plot on second image
        axes[j].plot(_x_data[:, 0], _x_data[:, 1])
        x_data.append(_x_data)

    print(f'Function value in optimal point (minimum), f* = {quad(q, b, x_star)}')

    # additional build extra plots for convergence rates
    _, axes = plt.subplots(4, 2, figsize=(15, 10))
    plt.tight_layout()
    # grids on all subplots
    for j in range(2):
        for k in range(4):
            axes[k][j].grid()

    for j, data in enumerate([first, second]):
        # residual f(x_k) - f* (with broadcasting)
        f_k_residual = data['f_k'] - quad(q, b, x_star)
        N = len(f_k_residual)  # noqa

        # calculate residuals for vectors x_k - x* by two ways:
        # 1 - by dubbing x* and subtracting arrays of the same size
        x_k_residual = x_data[j] - np.kron(np.ones((N, 1)), x_star)
        # 2 - by broadcasting
        x_k_residual_copy = x_data[j] - np.atleast_2d(x_star)
        # check identity of the arrays
        np.testing.assert_array_equal(x_k_residual_copy, x_k_residual)

        axes[0][j].set_title('Function residual')
        axes[0][j].plot(f_k_residual, )
        axes[1][j].set_title('Logarithmic function residual')
        axes[1][j].plot(np.log(f_k_residual), )
        axes[2][j].set_title('Value residual (Euclidean norm)')
        axes[2][j].plot(np.arange(N), np.linalg.norm(x_k_residual, axis=1))
        axes[3][j].set_title('Logarithmic value residual (Euclidean norm)')
        axes[3][j].plot(np.arange(N), np.log(np.linalg.norm(x_k_residual, axis=1)))

    plt.show()


def draw_rosen(first, second):
    # plot Rosenbrock contours
    x, y = np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-2, 3, 400))  # noqa
    z_rosen = rosen2d_xy(x, y)

    # minimum
    f_star = 0
    x_star = np.array([1, 1])

    _, axes = plt.subplots(1, 2, figsize=(18, 10))
    for k in range(2):
        axes[k].plot([1], [1], 'x')
        axes[k].contour(x, y, z_rosen, np.logspace(-1, 3, 10))

    for i, data in enumerate([first, second]):
        x_data = np.array(data['x_k'])
        axes[i].set_title('Contours of Rosenbrock function')
        axes[i].plot(x_data[:, 0], x_data[:, 1], )

    # residuals
    _, axes = plt.subplots(4, 2, figsize=(15, 10))
    plt.tight_layout()

    # for both columns:
    for j, data in enumerate([first, second]):
        # draw grid in all sub-plots
        for m in range(4):
            axes[m][j].grid()

        x_data = np.array(data['x_k'])
        f_k_residual = np.array(data['f_k']) - f_star

        N = len(f_k_residual)  # noqa
        # broadcasting!
        x_k_residual = x_data - x_star

        axes[0][j].set_title('Function residual')
        axes[0][j].plot(f_k_residual, )
        axes[1][j].set_title('Logarithmic function residual')
        axes[1][j].plot(np.log(f_k_residual), )
        axes[2][j].set_title('Value residual (Euclidean norm)')
        axes[2][j].plot(np.arange(N), np.linalg.norm(x_k_residual, axis=1))
        axes[3][j].set_title('Logarithmic value residual (Euclidean norm)')
        axes[3][j].plot(np.arange(N), np.log(np.linalg.norm(x_k_residual, axis=1)))

    plt.show()


def draw_lines(lines: List[Tuple[str, np.ndarray]]):
    for line in lines:
        plt.plot(line[1], label=line[0])
    plt.title(f'Function residual')
    plt.legend()
    plt.show()

    for line in lines:
        plt.plot(np.log(line[1]), label=line[0])
    plt.title(f'Logarithmic function residual')
    plt.legend()
    plt.show()


def draw_constr_line(q, b, a, d, x_0, x_star, data):
    # plot contours
    x, y = np.meshgrid(np.linspace(-5, 15, 400), np.linspace(-10, 3, 100))
    z_quad = quad_2d_xy(q, b, x, y)

    line_x2 = np.array([-10, 3])
    line_x1 = (-d - a[1] * line_x2) / a[0]

    # some step-size
    gamma = 0.6
    # levels for contour lines
    levels = quad(q, b, x_star) + np.linspace(0, 8, 12) ** 2

    _, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(x_star[0], x_star[1], '*')
    ax.plot(x_0[0], x_0[1], 'or')
    ax.plot((x_0[0], (x_0 - gamma * quad_grad(q, b, x_0))[0]),
            (x_0[1], (x_0 - gamma * quad_grad(q, b, x_0))[1]), 'r')
    ax.plot(line_x1, line_x2, 'g')
    ax.contour(x, y, z_quad, levels)
    ax.axis('equal')

    x_data = []
    # convert list of vectors to 2D-array of dimension N x 2
    _x_data = np.array(data['x_k'])
    # plot on second image
    ax.plot(_x_data[:, 0], _x_data[:, 1])
    x_data.append(_x_data)

    print(f'Function value in optimal point (minimum), f* = {quad(q, b, x_star)}')

    _, axes = plt.subplots(4, 1, figsize=(15, 10))
    plt.tight_layout()
    # grids on all subplots

    for m in range(4):
        axes[m].grid()

    x = x_data[0][-1]
    # residual f(x_k) - f* (with broadcasting)
    f_k_residual = data['f_k'] - quad(q, b, x)
    n = len(f_k_residual)
    # calculate residuals for vectors x_k - x* by two ways:
    # 1 - by dubbing x* and subtracting arrays of the same size
    x_k_residual = x_data[0] - np.kron(np.ones((n, 1)), x)
    # 2 - by broadcasting
    x_k_residual_copy = x_data[0] - np.atleast_2d(x)
    # check identity of the arrays
    np.testing.assert_array_equal(x_k_residual_copy, x_k_residual)

    axes[0].set_title('Function residual')
    axes[0].plot(f_k_residual)
    axes[1].set_title('Logarithmic function residual')
    axes[1].plot(np.log(abs(f_k_residual)))
    axes[2].set_title('Value residual (Euclidean norm)')
    axes[2].plot(np.arange(n), np.linalg.norm(x_k_residual, axis=1))
    axes[3].set_title('Logarithmic value residual (Euclidean norm)')
    axes[3].plot(np.arange(n), np.log(np.linalg.norm(x_k_residual, axis=1)))

    plt.show()


def draw_constr_circle(q, b, r, center, x_0, x_star, data):
    X, Y = np.meshgrid(np.linspace(-5, 15, 400), np.linspace(-10, 3, 100))  # noqa
    Z_quad = quad_2d_xy(q, b, X, Y)  # noqa

    # some step-size
    gamma = 0.6
    # levels for contour lines
    levels = quad(q, b, x_star) + np.linspace(0, 8, 12) ** 2

    # circle
    alpha = np.linspace(0, 2 * np.pi, 100)
    circle_x1 = r * np.cos(alpha) + center[0]
    circle_x2 = r * np.sin(alpha) + center[1]

    _, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.plot(x_star[0], x_star[1], '*')
    ax.plot(x_0[0], x_0[1], 'or')
    ax.plot((x_0[0], (x_0 - gamma * quad_grad(q, b, x_0))[0]),
            (x_0[1], (x_0 - gamma * quad_grad(q, b, x_0))[1]), 'r')
    ax.plot(circle_x1, circle_x2, 'g')
    ax.contour(X, Y, Z_quad, levels)
    ax.axis('equal')

    x_data = []
    # convert list of vectors to 2D-array of dimension N x 2
    _x_data = np.array(data['x_k'])
    # plot on second image
    ax.plot(_x_data[:, 0], _x_data[:, 1])
    x_data.append(_x_data)

    print(f'Function value in optimal point (minimum), f* = {quad(q, b, x_star)}')

    _, axes = plt.subplots(4, 1, figsize=(15, 10))
    plt.tight_layout()
    # grids on all subplots

    for m in range(4):
        axes[m].grid()

    x = x_data[0][-1]
    # residual f(x_k) - f* (with broadcasting)
    f_k_residual = data['f_k'] - quad(q, b, x)
    n = len(f_k_residual)
    # calculate residuals for vectors x_k - x* by two ways:
    # 1 - by dubbing x* and subtracting arrays of the same size
    x_k_residual = x_data[0] - np.kron(np.ones((n, 1)), x)
    # 2 - by broadcasting
    x_k_residual_copy = x_data[0] - np.atleast_2d(x)
    # check identity of the arrays
    np.testing.assert_array_equal(x_k_residual_copy, x_k_residual)

    axes[0].set_title('Function residual')
    axes[0].plot(f_k_residual)
    axes[1].set_title('Logarithmic function residual')
    axes[1].plot(np.log(abs(f_k_residual)))
    axes[2].set_title('Value residual (Euclidean norm)')
    axes[2].plot(np.arange(n), np.linalg.norm(x_k_residual, axis=1))
    axes[3].set_title('Logarithmic value residual (Euclidean norm)')
    axes[3].plot(np.arange(n), np.log(np.linalg.norm(x_k_residual, axis=1)))

    plt.show()
