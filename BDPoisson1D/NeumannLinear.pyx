from __future__ import division, print_function
import warnings
import numpy as np
from scipy.sparse import linalg, dia_matrix

from._helpers cimport trapz_1d


cpdef neumann_poisson_solver_arrays(double[:] nodes, double[:] f_nodes,
                                    double bc1, double bc2, double j=1.0, double y0=0.0):
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
    cdef:
        double integral
        double[:] a, b, c, y, f, dy, d2y, residual
    integral = trapz_1d(f_nodes, nodes)
    if abs(integral - bc2 + bc1) > 1e-4:
        warnings.warn('Not well-posed! Redefine f function and boundary conditions or refine the mesh!')
    step = np.array(nodes[1:]) - np.array(nodes[:-1])  # grid step
    a = -2 * np.ones(nodes.size - 1)
    a[0] = 0
    b = np.ones(nodes.size - 1)
    c = np.ones(nodes.size - 1)
    b[-2] = 2
    m = dia_matrix(([b, a, c], [-1, 0, 1]), (nodes.size - 1, nodes.size - 1), dtype=np.float).tocsr()
    f = (j * step) ** 2 * f_nodes[1:]
    f[0] += step[0] ** 2 * f_nodes[0] + 2 * step[0] * bc1 + y0
    f[-1] -= 2 * step[-1] * bc2
    y = linalg.spsolve(m, np.array(f), use_umfpack=True)
    y = np.append([y0], np.array(y))  # solution vector
    dy = np.gradient(y, nodes, edge_order=2) / j
    d2y = np.gradient(dy, nodes, edge_order=2) / j
    residual = np.array(f_nodes) - np.array(d2y)
    return np.array(y), np.array(residual)


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