from __future__ import division, print_function
import numpy as np
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import spsolve

from cython cimport boundscheck, wraparound

from BDMesh.Mesh1DUniform cimport Mesh1DUniform
from BDMesh.TreeMesh1DUniform cimport  TreeMesh1DUniform
from ._helpers cimport fd_d2_matrix, adjust_range, points_for_refinement
from .Function cimport Function, Functional, InterpolateFunction


@boundscheck(False)
@wraparound(False)
cpdef dirichlet_non_linear_poisson_solver_arrays(double[:] nodes, double[:] y0_nodes,
                                                 double[:] f_nodes, double[:] df_ddy_nodes,
                                                 double bc1, double bc2, double j=1.0, double w=1.0):
    """
    Solves 1D differential equation of the form
        d2y/dx2 = f(x, y(x))
        y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
    using FDE algorithm of O(h2) precision and Tailor series for linearization.
        y = y0 + Dy
        f(x, y(x)) ~= f(x, y0) + df/dDy(x, y0)*Dy
    :param nodes: 1D array of x nodes. Must include boundary points.
    :param y0_nodes: 1D array of y(x) initial approximation at nodes. Must be same shape as nodes.
    :param f_nodes: 1D array of values of f(x) on nodes array. Must be same shape as nodes.
    :param df_ddy_nodes: df/dDy, where Dy is delta y for y0 correction.
    :param bc1: boundary condition at nodes[0] point (a number).
    :param bc2: boundary condition at nodes[n] point (a number).
    :param j: Jacobian.
    :param w: the weight for Dy (default w=1.0).
    :return: solution y = y0 + w * Dy; Dy; residual.
    """
    cdef:
        int i, n = nodes.size
        double[:] step = np.zeros(n - 2, dtype=np.double)  # grid step
        double[:] dy = np.zeros(n, dtype=np.double)  # solution vector
        double[:] y = np.zeros(n, dtype=np.double)
        double[:] residual = np.zeros(n, dtype=np.double)
        double[:] a = np.zeros(n - 2, dtype=np.double)
        double[:] f = np.zeros(n - 2, dtype=np.double)
        double[:] b, sol
    m1 = fd_d2_matrix(n - 2)
    b = m1.dot(np.asarray(y0_nodes[1:n - 1]))
    for i in range(n - 2):
        step[i] = nodes[i + 1] - nodes[i]
        a[i] = (j * step[i]) ** 2 * df_ddy_nodes[i + 1]
        f[i] = (j * step[i]) ** 2 * f_nodes[i + 1] - b[i]
    m = m1 - dia_matrix(([a], [0]), (n - 2, n - 2)).tocsr()
    f[0] -= bc1
    f[n - 3] -= bc2
    dy[0] = bc1 - y0_nodes[0]
    y[0] = y0_nodes[0] + w * dy[0]
    dy[n - 1] = bc2 - y0_nodes[n - 1]
    y[n - 1] = y0_nodes[n - 1] + w * dy[n - 1]
    sol = spsolve(m, f, use_umfpack=True)
    for i in range(1, n - 1):
        dy[i] = sol[i - 1]
        y[i] = y0_nodes[i] + w * dy[i]
    d_y0 = np.gradient(y0_nodes, nodes, edge_order=2) / j
    d2_y0 = np.gradient(d_y0, nodes, edge_order=2) / j
    d_dy = np.gradient(dy, nodes, edge_order=2) / j
    d2_dy = np.gradient(d_dy, nodes, edge_order=2) / j
    for i in range(n):
        residual[i] = f_nodes[i] - d2_dy[i] - d2_y0[i]
    return np.asarray(y), np.asarray(dy), np.asarray(residual)


cpdef dirichlet_non_linear_poisson_solver(double[:] nodes, Function y0, Functional f, Functional df_ddy,
                                          double bc1, double bc2, double j=1.0, double w=1.0):
    """
    Solves 1D differential equation of the form
        d2y/dx2 = f(x, y(x))
        y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
    using FDE algorithm of O(h2) precision and Tailor series for linearization.
        y = y0 + Dy
        f(x, y(x)) ~= f(x, y0) + df/dDy(x, y0)*Dy
    :param nodes: 1D array of x nodes. Must include boundary points.
    :param y0: callable of y(x) initial approximation.
    :param f: callable of f(x) to be evaluated on nodes array.
    :param df_ddy: callable for evaluation of df/dDy, where Dy is delta y for y0 correction.
    :param bc1: boundary condition at nodes[0] point (a number).
    :param bc2: boundary condition at nodes[n] point (a number).
    :param j: Jacobian.
    :param w: the weight for Dy (default w=1.0).
    :return: solution as callable function y = y0 + w * Dy; Dy; residual.
    """
    cdef:
        double[:] y_nodes, dy, residual
        InterpolateFunction y
    y_nodes, dy, residual = dirichlet_non_linear_poisson_solver_arrays(nodes, y0.evaluate(nodes),
                                                                       f.evaluate(nodes), df_ddy.evaluate(nodes),
                                                                       bc1, bc2, j, w)
    y = InterpolateFunction(nodes, y_nodes)
    return y, np.asarray(dy), np.asarray(residual)


cpdef dirichlet_non_linear_poisson_solver_mesh_arrays(Mesh1DUniform mesh,
                                                      double[:] y0_nodes, double[:] f_nodes,
                                                      double[:] df_ddy_nodes, double w=1.0):
    """
    Solves 1D differential equation of the form
        d2y/dx2 = f(x, y(x))
        y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
    using FDE algorithm of O(h2) precision and Tailor series for linearization.
        y = y0 + Dy
        f(x, y(x)) ~= f(x, y0) + df/dDy(x, y0)*Dy
    :param mesh: 1D Uniform mesh with boundary conditions and Jacobian.
    :param y0_nodes: 1D array of y(x) initial approximation at mesh nodes.
    :param f_nodes: 1D array of values of f(x) on mesh nodes.
    :param df_ddy_nodes: df/dDy, where Dy is delta y for y0 correction.
    :param w: the weight for Dy (default w=1.0)
    :return: mesh with solution y = y0 + w * Dy, and residual; Dy.
    """
    cdef:
        double[:] y_nodes, dy, residual
    y_nodes, dy, residual = dirichlet_non_linear_poisson_solver_arrays(mesh.__local_nodes, y0_nodes, f_nodes,
                                                                       df_ddy_nodes,
                                                                       mesh.__boundary_condition_1,
                                                                       mesh.__boundary_condition_2,
                                                                       mesh.j(), w)
    mesh.solution = y_nodes
    mesh.residual = residual
    return mesh, dy


cpdef dirichlet_non_linear_poisson_solver_mesh(Mesh1DUniform mesh, Function y0, Functional f, Functional df_ddy,
                                               double w=1.0):
    """
    Solves 1D differential equation of the form
        d2y/dx2 = f(x, y(x))
        y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
    using FDE algorithm of O(h2) precision and Tailor series for linearization.
        y = y0 + Dy
        f(x, y(x)) ~= f(x, y0) + df/dDy(x, y0)*Dy
    :param mesh: 1D Uniform mesh with boundary conditions and Jacobian.
    :param y0: callable of y(x) initial approximation.
    :param f: callable of f(x) to be evaluated on nodes array.
    :param df_ddy: callable for evaluation of df/dDy, where Dy is delta y for y0 correction.
    :param w: the weight for Dy (default w=1.0)
    :return: mesh with solution y = y0 + w * Dy, and residual; callable solution function; Dy.
    """
    cdef:
        double[:] physical_nodes = mesh.to_physical_coordinate(mesh.__local_nodes)
        double[:] dy
    mesh, dy = dirichlet_non_linear_poisson_solver_mesh_arrays(mesh,
                                                               y0.evaluate(physical_nodes),
                                                               f.evaluate(physical_nodes),
                                                               df_ddy.evaluate(physical_nodes), w)
    y = InterpolateFunction(physical_nodes, mesh.__solution)
    return mesh, y, dy


@boundscheck(False)
@wraparound(False)
cpdef dirichlet_non_linear_poisson_solver_recurrent_mesh(Mesh1DUniform mesh,
                                                         Function y0, Functional f, Functional df_ddy,
                                                         int max_iter=1000, double threshold=1e-7):
    """
    Solves 1D differential equation of the form
        d2y/dx2 = f(x, y(x))
        y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
    using FDE algorithm of O(h2) precision and Tailor series for linearization.
        y = y0 + Dy
        f(x, y(x)) ~= f(x, y0) + df/dDy(x, y0)*Dy
    Recurrent successive approximation of y0 is used to achieve given residual error threshold value.
    :param mesh: 1D Uniform mesh with boundary conditions and Jacobian.
    :param y0: callable of y(x) initial approximation.
    :param f: callable of f(x) to be evaluated on nodes array.
    :param df_ddy: callable for evaluation of df/dDy, where Dy is delta y for y0 correction.
    :param max_iter: maximal number of allowed iterations.
    :param threshold: convergence residual error threshold.
    :return: mesh with solution y = y0 + w * Dy, and residual; callable solution function.
    """
    cdef double[:] dy
    for i in range(max_iter):
        mesh, y0, dy = dirichlet_non_linear_poisson_solver_mesh(mesh, y0, f, df_ddy)
        if abs(mesh.integrational_residual) <= threshold or max(abs(np.asarray(dy))) <= 2 * np.finfo(np.float).eps:
            break
        f.__f = y0
        df_ddy.__f = y0
    return mesh, y0

@boundscheck(False)
@wraparound(False)
cpdef dirichlet_non_linear_poisson_solver_amr(double boundary_1, double boundary_2, double step, Function y0,
                                              Functional f, Functional df_ddy, double bc1, double bc2,
                                              int max_iter=1000,
                                              double residual_threshold=1e-3, double int_residual_threshold=1e-6,
                                              int max_level=20, double mesh_refinement_threshold=1e-7):
    """
        Solves 1D differential equation of the form
            d2y/dx2 = f(x, y(x))
            y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
        using FDE algorithm of O(h2) precision and Tailor series for linearization.
            y = y0 + Dy
            f(x, y(x)) ~= f(x, y0) + df/dDy(x, y0)*Dy
        Recurrent successive approximation of y0 and adaptive mesh refinement
        algorithms are used to achieve given residual error threshold value.
        :param boundary_1: physical nodes left boundary.
        :param boundary_2: physical nodes right boundary.
        :param step: physical nodes step.
        :param y0: callable of y(x) initial approximation.
        :param f: callable of f(x) to be evaluated on nodes array.
        :param df_ddy: callable for evaluation of df/dDy, where Dy is delta y for y0 correction.
        :param bc1: boundary condition at nodes[0] point (a number).
        :param bc2: boundary condition at nodes[n] point (a number).
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
    while i < max_iter:
        i += 1
        level = max(meshes_tree.levels)
        converged = np.zeros(len(meshes_tree.__tree[level]), dtype=np.int)
        refinements = []
        for mesh_id, mesh in enumerate(meshes_tree.__tree[level]):
            mesh, y0 = dirichlet_non_linear_poisson_solver_recurrent_mesh(mesh, y0, f, df_ddy,
                                                                          max_iter, int_residual_threshold)
            f.__f = y0
            df_ddy.__f = y0
            mesh.trim()
            converged[mesh_id] = max(abs(np.asarray(mesh.residual))) < residual_threshold
            if converged[mesh_id]:
                continue
            refinement_points_chunks = points_for_refinement(mesh, mesh_refinement_threshold)
            converged[mesh_id] = not len(refinement_points_chunks) > 0
            if converged[mesh_id]:
                continue
            elif level < max_level:
                for block in refinement_points_chunks:
                    idx1, idx2, mesh_crop = adjust_range(block, mesh.__num - 1, crop=[10, 10],
                                                         step_scale=meshes_tree.refinement_coefficient)
                    refinements.append(Mesh1DUniform(
                        mesh.to_physical_coordinate(np.array([mesh.__local_nodes[idx1]]))[0],
                        mesh.to_physical_coordinate(np.array([mesh.__local_nodes[idx2]]))[0],
                        boundary_condition_1=mesh.__solution[idx1],
                        boundary_condition_2=mesh.__solution[idx2],
                        physical_step=mesh.physical_step/meshes_tree.refinement_coefficient,
                        crop=mesh_crop))
        meshes_tree.remove_coarse_duplicates()
        if np.asarray(converged).all() or level == max_level:
            break
        for refinement_mesh in refinements:
            meshes_tree.add_mesh(refinement_mesh)
    return meshes_tree
