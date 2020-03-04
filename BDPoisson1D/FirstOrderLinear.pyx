import numpy as np

from cython cimport boundscheck, wraparound
from cpython.array cimport array, clone

from scipy.linalg.cython_lapack cimport dgtsv

from BDMesh.Mesh1DUniform cimport Mesh1DUniform
from ._helpers cimport gradient1d
from .Function cimport Function


@boundscheck(False)
@wraparound(False)
cpdef double[:] dirichlet_first_order_solver_arrays(double[:] nodes, double[:] p_nodes, double[:] f_nodes,
                                                    double bc1, double bc2, double j=1.0):
    """
    Solves linear 1D differential equation of the form
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
        array[double] result, f, d, dl, du, template = array('d')
    result = clone(template, n, zero=False)
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
    d[0] += 1
    f[0] += + j * (nodes[1] - nodes[0]) * (f_nodes[0] - p_nodes[0] * bc1) + 2 * bc1
    d[nn - 1] += 1
    f[nn - 1] += - j * (nodes[n - 1] - nodes[n - 2]) * (f_nodes[n - 1] - p_nodes[n - 1] * bc2)
    dgtsv(&nn, &nrhs, &dl[0], &d[0], &du[0], &f[0], &nn, &info)
    result[0] = bc1
    result[n - 1] = bc2
    for i in range(nn):
        result[i + 1] = f[i]
    return result


@boundscheck(False)
@wraparound(False)
cpdef double[:] dirichlet_first_order_solver(double[:] nodes, Function p, Function f,
                                             double bc1, double bc2, double j=1.0):
    """
    Solves linear 1D differential equation of the form
        dy/dx + p(x)*y = f(x)
        y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
    using FDE algorithm of O(h2) precision.

    :param nodes: 1D array of x nodes. Must include boundary points.
    :param p: function p(x) callable on nodes array.
    :param f: function f(x) callable on nodes array.
    :param bc1: boundary condition at nodes[0] point (a number).
    :param bc2: boundary condition at nodes[n] point (a number).
    :param j: Jacobian.
    :return:
        y: 1D array of solution function y(x) values on nodes array.
        residual: error of the solution.
    """
    return dirichlet_first_order_solver_arrays(nodes, p.evaluate(nodes), f.evaluate(nodes), bc1, bc2, j)


@boundscheck(False)
@wraparound(False)
cpdef void dirichlet_first_order_solver_mesh_arrays(Mesh1DUniform mesh, double[:] p_nodes, double[:] f_nodes):
    """
    Solves linear 1D differential equation of the form
        dy/dx + p(x)*y = f(x)
        y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
    using FDE algorithm of O(h2) precision.

    :param mesh: BDMesh to solve on.
    :param p_nodes: 1D array of values of p(x) on nodes array. Must be same shape as nodes.
    :param f_nodes: 1D array of values of f(x) on nodes array. Must be same shape as nodes.
    """
    mesh.solution = dirichlet_first_order_solver_arrays(mesh.__local_nodes, p_nodes, f_nodes,
                                                        mesh.__boundary_condition_1, mesh.__boundary_condition_2,
                                                        mesh.j())


@boundscheck(False)
@wraparound(False)
cpdef void dirichlet_first_order_solver_mesh(Mesh1DUniform mesh, Function p, Function f):
    """
    Solves linear 1D differential equation of the form
        dy/dx + p(x)*y = f(x)
        y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
    using FDE algorithm of O(h2) precision.

    :param mesh: BDMesh to solve on.
    :param p: function p(x) callable on nodes array.
    :param f: function f(x) callable on nodes array.
    """
    dirichlet_first_order_solver_mesh_arrays(mesh, p.evaluate(mesh.physical_nodes), f.evaluate(mesh.physical_nodes))
