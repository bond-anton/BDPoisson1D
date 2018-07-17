from __future__ import division, print_function
import warnings
import numpy as np
from scipy.sparse import linalg, dia_matrix

from cython cimport boundscheck, wraparound

from._helpers cimport trapz_1d
from .Function cimport Function


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
        int i, n = nodes.size
        double integral = trapz_1d(f_nodes, nodes)
        double[:] y = np.zeros(n, dtype=np.double)
        double[:] residual = np.zeros(n, dtype=np.double)
        double[:] step = np.zeros(n - 1, dtype=np.double)
        double[:] f = np.zeros(n - 1, dtype=np.double)
        double[:] a = -2 * np.ones(n - 1, dtype=np.double)
        double[:] b = np.ones(n - 1, dtype=np.double)
        double[:] c = np.ones(n - 1, dtype=np.double)
        double[:] dy, d2y, sol
    if abs(integral - bc2 + bc1) > 1e-4:
        warnings.warn('Not well-posed! Redefine f function and boundary conditions or refine the mesh!')
    with boundscheck(False), wraparound(False):
        for i in range(n - 1):
            step[i] = nodes[i + 1] - nodes[i]  # grid step
            f[i] = (j * step[i]) ** 2 * f_nodes[i + 1]
    f[0] += step[0] ** 2 * f_nodes[0] + 2 * step[0] * bc1 + y0
    f[-1] -= 2 * step[-1] * bc2
    a[0] = 0
    b[-2] = 2
    m = dia_matrix(([b, a, c], [-1, 0, 1]), (n - 1, n - 1), dtype=np.double).tocsr()
    sol = linalg.spsolve(m, f, use_umfpack=True)
    with boundscheck(False), wraparound(False):
        for i in range(n - 1):
            y[i + 1] = sol[i]
    y[0] = y0
    dy = np.gradient(y, nodes, edge_order=2) / j
    d2y = np.gradient(dy, nodes, edge_order=2) / j
    with boundscheck(False), wraparound(False):
        for i in range(n):
            residual[i] = f_nodes[i] - d2y[i]
    return np.array(y), np.array(residual)


cpdef neumann_poisson_solver(double[:] nodes, Function f, double bc1, double bc2, double j=1.0, double y0=0.0):
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
    return neumann_poisson_solver_arrays(nodes, f.evaluate(nodes), bc1, bc2, j, y0)
