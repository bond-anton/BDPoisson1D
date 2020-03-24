import numpy as np

from cython cimport boundscheck, wraparound
from cpython.array cimport array, clone

from BDMesh.Mesh1DUniform cimport Mesh1DUniform
from BDFunction1D cimport Function
from BDFunction1D.Functional cimport Functional
from BDFunction1D.Interpolation cimport InterpolateFunction
from ._helpers cimport mean_square_root, gradient1d
from .FirstOrderLinear cimport dirichlet_first_order_solver_arrays


@boundscheck(False)
@wraparound(False)
cpdef double[:, :] dirichlet_non_linear_first_order_solver_arrays(double[:] nodes, double[:] y0_nodes,
                                                                  double[:] p_nodes,
                                                                  double[:] f_nodes, double[:] df_dy_nodes,
                                                                  double bc1, double bc2, double j=1.0, double w=1.0):
    """
    Solves nonlinear 1D differential equation of the form
        dy/dx + p(x)*y = f(x, y)
        y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
    using FDE algorithm of O(h2) precision and Tailor series for linearization.
        y = y0 + Dy
        f(x, y(x)) ~= f(x, y=y0) + df/dy(x, y=y0)*Dy
    ODE transforms to linear ODE for Dy 
        dDy/dx + (p(x) - df/dy(x, y=y0))*Dy = f(x, y0) - p(x)*y0(x) - dy0/dx

    :param nodes: 1D array of x nodes. Must include boundary points.
    :param y0_nodes: 1D array of y(x) initial approximation at nodes. Must be same shape as nodes.
    :param p_nodes: 1D array of values of p(x) on nodes array. Must be same shape as nodes.
    :param f_nodes: 1D array of values of f(x, y=y0) on nodes array. Must be same shape as nodes.
    :param df_dy_nodes: 1D array of values of df_dy(x, y=y0) on nodes array. Must be same shape as nodes.
    :param bc1: boundary condition at nodes[0] point (a number).
    :param bc2: boundary condition at nodes[0] point (a number).
    :param j: Jacobian.
    :param w: Weight of Dy.
    :return:
        result: solution y = y0 + w * Dy; Dy.
    """
    cdef:
        int i, n = nodes.shape[0], nrhs = 1, info
        double bc1_l, bc2_l
        double[:] result_l, dy, dy0
        array[double] fl, pl, template = array('d')
        double[:, :] result = np.empty((n, 2), dtype=np.double)
    fl = clone(template, n, zero=False)
    pl = clone(template, n, zero=False)
    dy0 = gradient1d(y0_nodes, nodes)
    for i in range(n):
        fl[i] = f_nodes[i] - dy0[i] / j - p_nodes[i] * y0_nodes[i]
        pl[i] = p_nodes[i] - df_dy_nodes[i]
    bc1_l = bc1 - y0_nodes[0]
    bc2_l = bc2 - y0_nodes[n - 1]
    result_l = dirichlet_first_order_solver_arrays(nodes, pl, fl, bc1_l, bc2_l, j)
    for i in range(n):
        result[i, 0] = y0_nodes[i] + w * result_l[i]
        result[i, 1] = result_l[i]
    return result


@boundscheck(False)
@wraparound(False)
cpdef double[:, :] dirichlet_non_linear_first_order_solver(double[:] nodes, Function y0, Function p,
                                                           Functional f, Functional df_dy,
                                                           double bc1, double bc2, double j=1.0, double w=1.0):
    """
    Solves nonlinear 1D differential equation of the form
        dy/dx + p(x)*y = f(x, y)
        y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
    using FDE algorithm of O(h2) precision and Tailor series for linearization.
        y = y0 + Dy
        f(x, y(x)) ~= f(x, y=y0) + df/dy(x, y=y0)*Dy
    ODE transforms to linear ODE for Dy 
        dDy/dx + (p(x) - df/dy(x, y=y0))*Dy = f(x, y0) - p(x)*y0(x) - dy0/dx

    :param nodes: 1D array of x nodes. Must include boundary points.
    :param y0: initial approximation of function y(x).
    :param p: function p(x) callable on nodes array.
    :param f: function f(x) callable on nodes array.
    :param df_dy: function df/dy(x, y=y0) callable on nodes array.
    :param bc1: boundary condition at nodes[0] point (a number).
    :param bc2: boundary condition at nodes[n] point (a number).
    :param j: Jacobian.
    :param w: Weight of Dy.
    :return:
        result: solution y = y0 + w * Dy; Dy.
    """
    return dirichlet_non_linear_first_order_solver_arrays(nodes, y0.evaluate(nodes), p.evaluate(nodes),
                                                          f.evaluate(nodes), df_dy.evaluate(nodes),
                                                          bc1, bc2, j, w)


@boundscheck(False)
@wraparound(False)
cpdef void dirichlet_non_linear_first_order_solver_mesh_arrays(Mesh1DUniform mesh, double[:] y0_nodes,
                                                               double[:] p_nodes,
                                                               double[:] f_nodes, double[:] df_dy_nodes, double w=1.0):
    """
    Solves nonlinear 1D differential equation of the form
        dy/dx + p(x)*y = f(x, y)
        y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
    using FDE algorithm of O(h2) precision and Tailor series for linearization.
        y = y0 + Dy
        f(x, y(x)) ~= f(x, y=y0) + df/dy(x, y=y0)*Dy
    ODE transforms to linear ODE for Dy 
        dDy/dx + (p(x) - df/dy(x, y=y0))*Dy = f(x, y0) - p(x)*y0(x) - dy0/dx
        
    :param mesh: BDMesh to solve on.
    :param y0_nodes: 1D array of y(x) initial approximation at nodes. Must be same shape as nodes.
    :param p_nodes: 1D array of values of p(x) on nodes array. Must be same shape as nodes.
    :param f_nodes: 1D array of values of f(x, y=y0) on nodes array. Must be same shape as nodes.
    :param df_dy_nodes: 1D array of values of df_dy(x, y=y0) on nodes array. Must be same shape as nodes.
    :param w: Weight of Dy.
    """
    cdef:
        double[:, :] result
    result = dirichlet_non_linear_first_order_solver_arrays(mesh.__local_nodes, y0_nodes, p_nodes,
                                                            f_nodes, df_dy_nodes,
                                                            mesh.__boundary_condition_1, mesh.__boundary_condition_2,
                                                            mesh.j(), w)
    mesh.solution = result[:, 0]
    mesh.residual = result[:, 1]


@boundscheck(False)
@wraparound(False)
cpdef void dirichlet_non_linear_first_order_solver_mesh(Mesh1DUniform mesh, Function y0, Function p,
                                                        Functional f, Functional df_dy,
                                                        double w=1.0):
    """
    Solves nonlinear 1D differential equation of the form
        dy/dx + p(x)*y = f(x, y)
        y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
    using FDE algorithm of O(h2) precision and Tailor series for linearization.
        y = y0 + Dy
        f(x, y(x)) ~= f(x, y=y0) + df/dy(x, y=y0)*Dy
    ODE transforms to linear ODE for Dy 
        dDy/dx + (p(x) - df/dy(x, y=y0))*Dy = f(x, y0) - p(x)*y0(x) - dy0/dx

    :param mesh: BDMesh to solve on.
    :param y0: initial approximation of function y(x).
    :param p: function p(x) callable on nodes array.
    :param f: function f(x) callable on nodes array.
    :param df_dy: function df/dy(x, y=y0) callable on nodes array.
    :param w: Weight of Dy.
    """
    dirichlet_non_linear_first_order_solver_mesh_arrays(mesh, y0.evaluate(mesh.physical_nodes),
                                                        p.evaluate(mesh.physical_nodes),
                                                        f.evaluate(mesh.physical_nodes),
                                                        df_dy.evaluate(mesh.physical_nodes), w)


@boundscheck(False)
@wraparound(False)
cpdef void dirichlet_non_linear_first_order_solver_recurrent_mesh(Mesh1DUniform mesh, Function y0, Function p,
                                                                  Functional f, Functional df_dy, double w=0.0,
                                                                  int max_iter=1000, double threshold=1e-7):
    """
    Solves nonlinear 1D differential equation of the form
        dy/dx + p(x)*y = f(x, y)
        y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
    using FDE algorithm of O(h2) precision and Tailor series for linearization.
        y = y0 + Dy
        f(x, y(x)) ~= f(x, y=y0) + df/dy(x, y=y0)*Dy
    ODE transforms to linear ODE for Dy 
        dDy/dx + (p(x) - df/dy(x, y=y0))*Dy = f(x, y0) - p(x)*y0(x) - dy0/dx

    :param mesh: BDMesh to solve on.
    :param y0: callable of y(x) initial approximation.
    :param p: function p(x) callable on nodes array.
    :param f: callable of f(x) to be evaluated on nodes array.
    :param df_dy: function df/dy(x, y=y0) callable on nodes array.
    :param w: Weight of Dy in the range [0.0..1.0]. If w=0.0 weight is set automatically.
    :param max_iter: maximal number of allowed iterations.
    :param threshold: convergence residual error threshold.
    :return: mesh with solution y = y0 + w * Dy, and Dy as a residual; callable solution function.
    """
    cdef:
        int i
        bint auto
        double res, res_old = 1e100, min_w = 0.3
    if w <= 0:
        auto = True
        w = 1.0
    elif w >= 1.0:
        auto = False
        w = 1.0
    else:
        auto = False
    i = 0
    while i < max_iter:
        dirichlet_non_linear_first_order_solver_mesh(mesh, y0, p, f, df_dy, w)
        y0 = InterpolateFunction(mesh.to_physical_coordinate(mesh.__local_nodes), mesh.__solution)
        f.__f = y0
        df_dy.__f = y0
        res = mean_square_root(mesh.residual)
        if res <= threshold:
            break
        if auto:
            if res > res_old:
                if w > min_w:
                    w -= 0.1
                    continue
                else:
                    break
            res_old = res
        i += 1
