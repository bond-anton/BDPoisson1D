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
    d[0] += 1
    f[0] += + j * (nodes[1] - nodes[0]) * (f_nodes[0] - p_nodes[0] * bc1) + 2 * bc1
    d[nn - 1] += 1
    f[nn - 1] += - j * (nodes[n - 1] - nodes[n - 2]) * (f_nodes[n - 1] - p_nodes[n - 1] * bc2)
    # f[nn - 1] += j * (nodes[n - 1] - nodes[n - 2]) * (f_nodes[n - 1] - p_nodes[n - 1] * bc2) - 2 * bc2
    dgtsv(&nn, &nrhs, &dl[0], &d[0], &du[0], &f[0], &nn, &info)
    result[0, 0] = bc1
    result[n - 1, 0] = bc2
    for i in range(nn):
        result[i + 1, 0] = f[i]
    dy = gradient1d(result[:, 0], nodes)
    for i in range(n):
        result[i, 1] = f_nodes[i] - dy[i] / j - p_nodes[i] * result[i, 0]
    return result


@boundscheck(False)
@wraparound(False)
cpdef double[:, :] dirichlet_first_order_solver(double[:] nodes, Function p, Function f,
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
    cdef:
        double[:, :] result
    result = dirichlet_first_order_solver_arrays(mesh.__local_nodes, p_nodes, f_nodes,
                                                 mesh.__boundary_condition_1, mesh.__boundary_condition_2,
                                                 mesh.j())
    mesh.solution = result[:, 0]
    mesh.residual = result[:, 1]


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


@boundscheck(False)
@wraparound(False)
cpdef void dirichlet_first_order_solver_mesh_amr(TreeMesh1DUniform meshes_tree, Function p, Function f,
                                                 int max_iter=1000, double threshold=1e-2, int max_level=10):
    """
    Linear first order ODE equation solver with Adaptive Mesh Refinement algorithm.
    :param meshes_tree: mesh_tree to start with (only root mesh is needed).
    :param p: function p(x) callable on nodes array.
    :param f: function f(x) callable on nodes array.
    :param max_iter: maximal number of allowed iterations.
    :param threshold: algorithm convergence residual threshold value.
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
            dirichlet_first_order_solver_mesh(mesh, p, f)
            mesh.trim()
            refinements = refinement_points(mesh, threshold, crop_l=20, crop_r=20,
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
cpdef TreeMesh1DUniform dirichlet_first_order_solver_amr(double boundary_1, double boundary_2, double step,
                                                         Function p, Function f,
                                                         double bc1, double bc2,
                                                         int max_iter=1000,
                                                         double threshold=1e-2, int max_level=10):
    """
    Linear first order ODE equation solver with Adaptive Mesh Refinement algorithm.
    :param boundary_1: physical nodes left boundary.
    :param boundary_2: physical nodes right boundary.
    :param step: physical nodes step.
    :param p: function p(x) callable on nodes array.
    :param f: function f(x) callable on nodes array.
    :param bc1: boundary condition at nodes[0] point (a number).
    :param bc2: boundary condition at nodes[n] point (a number).
    :param max_iter: maximal number of allowed iterations.
    :param threshold: algorithm convergence residual threshold value.
    :param max_level: max level of mesh refinement.
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
    dirichlet_first_order_solver_mesh_amr(meshes_tree, p, f, max_iter, threshold, max_level)
    return meshes_tree
