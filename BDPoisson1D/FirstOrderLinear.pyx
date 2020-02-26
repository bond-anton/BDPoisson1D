import numpy as np

from cython cimport boundscheck, wraparound
from cpython.array cimport array, clone

from scipy.linalg.cython_lapack cimport dgtsv

from BDMesh.TreeMesh1DUniform cimport TreeMesh1DUniform
from BDMesh.Mesh1DUniform cimport Mesh1DUniform
from ._helpers cimport gradient1d, refinement_points
from .Function cimport Function


@boundscheck(False)
@wraparound(False)
cpdef double[:, :] dirichlet_first_order_solver_arrays(double[:] nodes, double[:] p_nodes, double[:] f_nodes,
                                                        double bc1, double bc2, double j=1.0):
    """
    Solves 1D differential equation of the form
        dy/dx + p(x)*y = f(x)
        y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
    using FDE algorithm of O(h2) precision.

    :param nodes: 1D array of x nodes. Must include boundary points.
    :param p_nodes: 1D array of values of p(x) on nodes array. Must be same shape as nodes.
    :param f_nodes: 1D array of values of f(x) on nodes array. Must be same shape as nodes.
    :param bc1: boundary condition at nodes[0] point (a number).
    :param bc2: boundary condition at nodes[0] point (a number).
    :param j: Jacobian.
    :return:
        result: 2D array of solution function y(x) values on nodes array and the error of the solution.
    """
    cdef:
        int i, n = nodes.shape[0], nrhs = 1, info, nn
        double[:] dy
        array[double] f, d, dl, du, template = array('d')
        double[:, :] result = np.empty((n, 2), dtype=np.double)
    nn = n - 2
    d = clone(template, nn, zero=False)
    dl = clone(template, nn - 1, zero=False)
    du = clone(template, nn - 1, zero=False)
    f = clone(template, nn, zero=False)
    for i in range(nn):
        if i < nn - 1:
            dl[i] = -1.0
            du[i] = 1.0
        d[i] = j * (nodes[i + 2] - nodes[i]) * p_nodes[i + 1]
        f[i] = j * (nodes[i + 2] - nodes[i]) * f_nodes[i + 1]
    d[0] = d[0] + 1
    d[nn - 1] = d[nn - 1] + 1
    f[0]  = f[0] + j * (nodes[1] - nodes[0]) * (f_nodes[0] - p_nodes[0] * bc1) + 2 * bc1
    f[nn - 1]  = f[nn - 1] - j * (nodes[n - 1] - nodes[n - 2]) * (f_nodes[n - 1] - p_nodes[n - 1] * bc2)
    dgtsv(&nn, &nrhs, &dl[0], &d[0], &du[0], &f[0], &nn, &info)
    result[0, 0] = bc1
    result[n - 1, 0] = bc2
    for i in range(nn):
        result[i + 1, 0] = f[i]
    dy = gradient1d(result[:, 0], nodes, n)
    for i in range(n):
        result[i, 1] = f_nodes[i] - dy[i] / j - p_nodes[i] * result[i, 0]
    return result
