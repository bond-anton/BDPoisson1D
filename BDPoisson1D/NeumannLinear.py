from __future__ import division, print_function

import numpy as np
from scipy.sparse import linalg

from ._helpers import fd_d2_matrix


def neumann_poisson_solver_arrays(nodes, f_nodes, bc1, bc2, j=1, y0=0):
    """
    Solves 1D differential equation of the form
        d2y/dx2 = f(x)
        dy/dx(x0) = bc1, dy/dx(xn) = bc2 (Neumann boundary condition)
    using FDE algorithm of O(h2) precision.

    :param nodes: 1D array of x nodes. Must include boundary points.
    :param f_nodes: 1D array of values of f(x) on nodes array. Must be same shape as nodes.
    :param bc1: boundary condition at nodes[0] point (a number).
    :param bc2: boundary condition at nodes[0] point (a number).
    :param j: Jacobian.
    :param y0: value of y(x) at point x0 of nodes array
    :return:
        y: 1D array of solution function y(x) values on nodes array.
        residual: error of the solution.
    """
    integral = np.trapz(f_nodes, nodes)
    if abs(integral - bc2 + bc1) > 1e-4:
        print('WARNING!!!!')
        print('The problem is not well-posed!')
        print('Redefine the f function and BCs or refine the mesh!')
        print('WARNING!!!!')
    step = nodes[1:] - nodes[:-1]  # grid step
    m = fd_d2_matrix(nodes.size - 1)
    m[0, 0] = 0
    m[-1, -2] = 2
    y = np.append([y0], np.zeros(nodes.size - 1))  # solution vector
    f = (j * step) ** 2 * f_nodes[1:]
    f[0] += step[0] ** 2 * f_nodes[0] + 2 * step[0] * bc1 + y0
    f[-1] -= 2 * step[-1] * bc2
    y[1:] = linalg.spsolve(m, f, use_umfpack=True)
    dy = np.gradient(y, nodes, edge_order=2) / j
    d2y = np.gradient(dy, nodes, edge_order=2) / j
    residual = f_nodes - d2y
    return y, residual


def neumann_poisson_solver(nodes, f, bc1, bc2, j=1, y0=0):
    """
    Solves 1D differential equation of the form
        d2y/dx2 = f(x)
        dy/dx(x0) = bc1, dy/dx(xn) = bc2 (Neumann boundary condition)
    using FDE algorithm of O(h2) precision.

    :param nodes: 1D array of x nodes. Must include boundary points.
    :param f: function f(x) callable on nodes array..
    :param bc1: boundary condition at nodes[0] point (a number).
    :param bc2: boundary condition at nodes[0] point (a number).
    :param j: Jacobian.
    :param y0: value of y(x) at point x0 of nodes array
    :return:
        y: 1D array of solution function y(x) values on nodes array.
        residual: error of the solution.
    """
    return neumann_poisson_solver_arrays(nodes, f(nodes), bc1, bc2, j, y0)
