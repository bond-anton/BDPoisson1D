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
                                                       double ic, double j=1.0):
    """
    Solves 1D differential equation of the form
        dy/dx + p(x)*y = f(x)
        y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
    using FDE algorithm of O(h2) precision.

    :param nodes: 1D array of x nodes. Must include boundary points.
    :param p_nodes: 1D array of values of p(x) on nodes array. Must be same shape as nodes.
    :param f_nodes: 1D array of values of f(x) on nodes array. Must be same shape as nodes.
    :param ic: initial condition at nodes[0] point (a number).
    :param j: Jacobian.
    :return:
        result: 2D array of solution function y(x) values on nodes array and the error of the solution.
    """
    cdef:
        int i, n = nodes.shape[0], nrhs = 1, info
        double[:] dy
        array[double] f, d, dl, du, template = array('d')
        double[:, :] result = np.empty((n, 2), dtype=np.double)
    d = clone(template, n, zero=False)
    dl = clone(template, n - 1, zero=False)
    du = clone(template, n - 1, zero=False)
    f = clone(template, n, zero=False)
    d[0] = -1 + j * (nodes[1] - nodes[0]) * p_nodes[0]
    dl[0] = -1.0
    du[0] = 1.0
    f[0] = j * (nodes[1] - nodes[0]) * f_nodes[0]
    d[n - 1] = 1 + j * (nodes[n - 1] - nodes[n - 2]) * p_nodes[n - 1]
    dl[n - 1] = -1.0
    du[n - 1] = 1.0
    f[n - 1] = j * (nodes[n - 1] - nodes[n - 2]) * f_nodes[n - 2]
    for i in range(1, n - 1):
        dl[i] = -1.0
        du[i] = 1.0
        d[i] = j * (nodes[i + 1] - nodes[i - 1]) * p_nodes[i]
        f[i] = j * (nodes[i + 1] - nodes[i - 1]) * f_nodes[i]
    dgtsv(&n, &nrhs, &dl[0], &d[0], &du[0], &f[0], &n, &info)
    for i in range(n):
        result[i, 0] = f[i] + ic
    dy = gradient1d(result[:, 0], nodes, n)
    for i in range(n):
        result[i, 1] = f_nodes[i] - dy[i] / j - p_nodes[i] * result[i, 0]
    return result
