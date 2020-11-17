import matplotlib.pyplot as plt
import numpy as np

from src.linopt.utils import quad_2d_xy, quad, quad_grad


def draw(q, b, x_0, x_star, first, second):
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
    # plt.show() - not needed if %matplotlib magic is used

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
