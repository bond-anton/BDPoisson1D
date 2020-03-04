import numpy as np

from cython cimport boundscheck, wraparound
from cpython.array cimport array, clone

from BDMesh.TreeMesh1DUniform cimport TreeMesh1DUniform
from BDMesh.Mesh1DUniform cimport Mesh1DUniform
from ._helpers cimport mean_square, gradient1d, refinement_points
from .Function cimport Function, Functional, InterpolateFunction
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
        res = mean_square(mesh.residual)
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

@boundscheck(False)
@wraparound(False)
cpdef void dirichlet_non_linear_first_order_solver_mesh_amr(TreeMesh1DUniform meshes_tree, Function y0, Function p,
                                                            Functional f, Functional df_dy, double w=1.0,
                                                            int max_iter=1000,
                                                            double residual_threshold=1e-7,
                                                            double int_residual_threshold=1e-6,
                                                            int max_level=20, double mesh_refinement_threshold=1e-7):
    """
    Nonlinear first order ODE equation solver with Adaptive Mesh Refinement algorithm.
    :param meshes_tree: mesh_tree to start with (only root mesh is needed).
    :param y0: callable of y(x) initial approximation.
    :param p: function p(x) callable on nodes array.
    :param f: callable of f(x) to be evaluated on nodes array.
    :param df_dy: function df/dy(x, y=y0) callable on nodes array.
    :param w: Weight of Dy in the range [0.0..1.0]. If w=0.0 weight is set automatically.
    :param max_iter: maximal number of allowed iterations.
    :param residual_threshold: algorithm convergence residual threshold value.
    :param int_residual_threshold: algorithm convergence integral residual threshold value.
    :param mesh_refinement_threshold
    :param max_level: max level of mesh refinement.
    """
    cdef:
        int level, i = 0, j, converged, n
        Mesh1DUniform mesh
        int[:, :] refinements
    while i < max_iter:
        i += 1
        level = max(meshes_tree.levels)
        converged = 0
        n = 0
        for mesh in meshes_tree.__tree[level]:
            n += 1
            dirichlet_non_linear_first_order_solver_recurrent_mesh(mesh, y0, p, f, df_dy, w,
                                                                   max_iter, int_residual_threshold)
            y0 = InterpolateFunction(mesh.to_physical_coordinate(mesh.__local_nodes), mesh.__solution)
            f.__f = y0
            df_dy.__f = y0
            mesh.trim()
            refinements = refinement_points(mesh, residual_threshold, crop_l=20, crop_r=20,
                                            step_scale=meshes_tree.refinement_coefficient)
            if refinements.shape[0] == 0:
                converged += 1
                continue
            if level < max_level and i < max_iter:
                for j in range(refinements.shape[0]):
                    meshes_tree.add_mesh(Mesh1DUniform(
                        mesh.__physical_boundary_1 + mesh.j() * mesh.__local_nodes[refinements[j][0]],
                        mesh.__physical_boundary_1 + mesh.j() * mesh.__local_nodes[refinements[j][1]],
                        boundary_condition_1=mesh.__solution[refinements[j][0]],
                        boundary_condition_2=mesh.__solution[refinements[j][1]],
                        physical_step=mesh.physical_step/meshes_tree.refinement_coefficient,
                        crop=[refinements[j][2], refinements[j][3]]))
        meshes_tree.remove_coarse_duplicates()
        if converged == n or level == max_level:
            break


@boundscheck(False)
@wraparound(False)
cpdef TreeMesh1DUniform dirichlet_non_linear_first_order_solver_amr(double boundary_1, double boundary_2, double step,
                                                                    Function y0, Function p,
                                                                    Functional f, Functional df_dy,
                                                                    double bc1, double bc2, double w=1.0,
                                                                    int max_iter=1000,
                                                                    double residual_threshold=1e-3,
                                                                    double int_residual_threshold=1e-6,
                                                                    int max_level=20,
                                                                    double mesh_refinement_threshold=1e-7):
    """
    Nonlinear first order ODE equation solver with Adaptive Mesh Refinement algorithm.
    :param boundary_1: physical nodes left boundary.
    :param boundary_2: physical nodes right boundary.
    :param step: physical nodes step.
    :param y0: callable of y(x) initial approximation.
    :param p: function p(x) callable on nodes array.
    :param f: callable of f(x) to be evaluated on nodes array.
    :param df_dy: callable for evaluation of df/dDy, where Dy is delta y for y0 correction.
    :param bc1: boundary condition at nodes[0] point (a number).
    :param bc2: boundary condition at nodes[n] point (a number).
    :param w: Weight of Dy.
    :param max_iter: maximal number of allowed iterations.
    :param residual_threshold: convergence residual error threshold.
    :param int_residual_threshold: convergence integrational residual error threshold.
    :param max_level: maximal level of allowed mesh refinement.
    :param mesh_refinement_threshold: convergence residual error threshold for mesh refinement.
    :return: meshes tree with solution and residual.
    """
    cdef:
        Mesh1DUniform root_mesh, mesh
        TreeMesh1DUniform meshes_tree
        int level, mesh_id, idx1, idx2, i = 0
        long[:] converged, block
        list refinements, refinement_points_chunks, mesh_crop
    root_mesh = Mesh1DUniform(boundary_1, boundary_2,
                              boundary_condition_1=bc1,
                              boundary_condition_2=bc2,
                              physical_step=round(step, 9))
    meshes_tree = TreeMesh1DUniform(root_mesh, refinement_coefficient=2, aligned=True)
    dirichlet_non_linear_first_order_solver_mesh_amr(meshes_tree, y0, p, f, df_dy, w,
                                                     max_iter, residual_threshold, int_residual_threshold,
                                                     max_level, mesh_refinement_threshold)
    return meshes_tree
