import numpy as np

from cython cimport boundscheck, wraparound
from cpython.array cimport array, clone

from scipy.linalg.cython_lapack cimport dgtsv

from BDMesh.TreeMesh1DUniform cimport TreeMesh1DUniform
from BDMesh.Mesh1DUniform cimport Mesh1DUniform
from ._helpers cimport gradient1d, refinement_points
from .Function cimport Function, Functional
from .FirstOrderLinear cimport dirichlet_first_order_solver_arrays


@boundscheck(False)
@wraparound(False)
cpdef double[:, :] dirichlet_non_linear_first_order_solver_arrays(double[:] nodes, double[:] y0_nodes,
                                                                  double[:] p_nodes,
                                                                  double[:] f_nodes, double[:] df_dy_nodes,
                                                                  double bc1, double bc2, double j=1.0, double w=1.0):
    """
    Solves 1D differential equation of the form
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
        result: solution y = y0 + w * Dy; Dy; residual.
    """
    cdef:
        int i, n = nodes.shape[0], nrhs = 1, info
        double bc1_l, bc2_l
        double[:] dy, dy0
        array[double] fl, pl, template = array('d')
        double[:, :] result_l, result = np.empty((n, 3), dtype=np.double)
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
        result[i, 0] = y0_nodes[i] + w * result_l[i, 0]
        result[i, 1] = result_l[i, 0]
    dy = gradient1d(result[:, 0], nodes)
    for i in range(n):
        result[i, 2] = f_nodes[i] - dy[i] / j - p_nodes[i] * result[i, 0]
    return result

#
# @boundscheck(False)
# @wraparound(False)
# cpdef double[:, :] dirichlet_first_order_solver(double[:] nodes, Function p, Function f,
#                                                 double bc1, double bc2, double j=1.0):
#     """
#     Solves 1D differential equation of the form
#         dy/dx + p(x)*y = f(x)
#         y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
#     using FDE algorithm of O(h2) precision.
#
#     :param nodes: 1D array of x nodes. Must include boundary points.
#     :param p: function p(x) callable on nodes array.
#     :param f: function f(x) callable on nodes array.
#     :param bc1: boundary condition at nodes[0] point (a number).
#     :param bc2: boundary condition at nodes[n] point (a number).
#     :param j: Jacobian.
#     :return:
#         y: 1D array of solution function y(x) values on nodes array.
#         residual: error of the solution.
#     """
#     return dirichlet_first_order_solver_arrays(nodes, p.evaluate(nodes), f.evaluate(nodes), bc1, bc2, j)
#
#
# @boundscheck(False)
# @wraparound(False)
# cpdef void dirichlet_first_order_solver_mesh_arrays(Mesh1DUniform mesh, double[:] p_nodes, double[:] f_nodes):
#     """
#     Solves 1D differential equation of the form
#         d2y/dx2 = f(x)
#         y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
#     using FDE algorithm of O(h2) precision.
#
#     :param mesh: BDMesh to solve on.
#     :param p_nodes: 1D array of values of p(x) on nodes array. Must be same shape as nodes.
#     :param f_nodes: 1D array of values of f(x) on nodes array. Must be same shape as nodes.
#     """
#     cdef:
#         double[:, :] result
#     result = dirichlet_first_order_solver_arrays(mesh.__local_nodes, p_nodes, f_nodes,
#                                                  mesh.__boundary_condition_1, mesh.__boundary_condition_2,
#                                                  mesh.j())
#     mesh.solution = result[:, 0]
#     mesh.residual = result[:, 1]
#
#
# @boundscheck(False)
# @wraparound(False)
# cpdef void dirichlet_first_order_solver_mesh(Mesh1DUniform mesh, Function p, Function f):
#     """
#     Solves 1D differential equation of the form
#         d2y/dx2 = f(x)
#         y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
#     using FDE algorithm of O(h2) precision.
#
#     :param mesh: BDMesh to solve on.
#     :param p: function p(x) callable on nodes array.
#     :param f: function f(x) callable on nodes array.
#     """
#     dirichlet_first_order_solver_mesh_arrays(mesh, p.evaluate(mesh.physical_nodes), f.evaluate(mesh.physical_nodes))
#
#
# @boundscheck(False)
# @wraparound(False)
# cpdef void dirichlet_first_order_solver_mesh_amr(TreeMesh1DUniform meshes_tree, Function p, Function f,
#                                                  int max_iter=1000, double threshold=1e-2, int max_level=10):
#     """
#     Linear Poisson equation solver with Adaptive Mesh Refinement algorithm.
#     :param meshes_tree: mesh_tree to start with (only root mesh is needed).
#     :param p: function p(x) callable on nodes array.
#     :param f: function f(x) callable on nodes array.
#     :param max_iter: maximal number of allowed iterations.
#     :param threshold: algorithm convergence residual threshold value.
#     :param max_level: max level of mesh refinement.
#     """
#     cdef:
#         int level, i = 0, j, converged, n
#         Mesh1DUniform mesh
#         int[:, :] refinements
#     while i < max_iter:
#         i += 1
#         level = max(meshes_tree.levels)
#         converged = 0
#         n = 0
#         for mesh in meshes_tree.__tree[level]:
#             n += 1
#             dirichlet_first_order_solver_mesh(mesh, p, f)
#             mesh.trim()
#             refinements = refinement_points(mesh, threshold, crop_l=20, crop_r=20,
#                                             step_scale=meshes_tree.refinement_coefficient)
#             if refinements.shape[0] == 0:
#                 converged += 1
#                 continue
#             if level < max_level and i < max_iter:
#                 for j in range(refinements.shape[0]):
#                     meshes_tree.add_mesh(Mesh1DUniform(
#                         mesh.__physical_boundary_1 + mesh.j() * mesh.__local_nodes[refinements[j][0]],
#                         mesh.__physical_boundary_1 + mesh.j() * mesh.__local_nodes[refinements[j][1]],
#                         boundary_condition_1=mesh.__solution[refinements[j][0]],
#                         boundary_condition_2=mesh.__solution[refinements[j][1]],
#                         physical_step=mesh.physical_step/meshes_tree.refinement_coefficient,
#                         crop=[refinements[j][2], refinements[j][3]]))
#         meshes_tree.remove_coarse_duplicates()
#         if converged == n or level == max_level:
#             break
#
#
# @boundscheck(False)
# @wraparound(False)
# cpdef TreeMesh1DUniform dirichlet_first_order_solver_amr(double boundary_1, double boundary_2, double step,
#                                                          Function p, Function f,
#                                                          double bc1, double bc2,
#                                                          int max_iter=1000,
#                                                          double threshold=1e-2, int max_level=10):
#     """
#     Linear Poisson equation solver with Adaptive Mesh Refinement algorithm.
#     :param boundary_1: physical nodes left boundary.
#     :param boundary_2: physical nodes right boundary.
#     :param step: physical nodes step.
#     :param p: function p(x) callable on nodes array.
#     :param f: function f(x) callable on nodes array.
#     :param bc1: boundary condition at nodes[0] point (a number).
#     :param bc2: boundary condition at nodes[n] point (a number).
#     :param max_iter: maximal number of allowed iterations.
#     :param threshold: algorithm convergence residual threshold value.
#     :param max_level: max level of mesh refinement.
#     :return: meshes tree with solution and residual.
#     """
#     cdef:
#         Mesh1DUniform root_mesh, mesh
#         TreeMesh1DUniform meshes_tree
#         int level, mesh_id, idx1, idx2, i = 0
#         long[:] converged, block
#         list refinements, refinement_points_chunks, mesh_crop
#     root_mesh = Mesh1DUniform(boundary_1, boundary_2,
#                               boundary_condition_1=bc1,
#                               boundary_condition_2=bc2,
#                               physical_step=round(step, 9))
#     meshes_tree = TreeMesh1DUniform(root_mesh, refinement_coefficient=2, aligned=True)
#     dirichlet_first_order_solver_mesh_amr(meshes_tree, p, f, max_iter, threshold, max_level)
#     return meshes_tree
