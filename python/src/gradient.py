def quad(Q, b, x):
    return (0.5 * Q.dot(x) + b).dot(x)  # or x.T @ (0.5 * Q.dot(x) + b)


def quad_grad(Q, b, x):
    return Q.dot(x) + b


def const_step_solver(x0, f, f_grad, params=None):
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
    x_array = [x]
    f_array = [f(x)]
    gamma = 0.55
    # stopping rule
    k = 0
    while k < 70:
        x = x + gamma * (-f_grad(x))
        x_array += [x]
        f_array += [f(x)]
        # print(f(x))
        k += 1
    print(f'{k} iterations is taken, f(x^k) = {f_array[-1]}')
    return {'x_k': x_array, 'f_k': f_array}
