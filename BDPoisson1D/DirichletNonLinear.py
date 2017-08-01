from __future__ import division, print_function

import time

import numpy as np
from scipy.interpolate import interp1d
from scipy import sparse
from scipy.sparse import linalg

from BDMesh import Mesh1DUniform, TreeMesh1DUniform
from ._helpers import fd_d2_matrix, interp_fn, adjust_range, points_for_refinement


def dirichlet_non_linear_poisson_solver_arrays(nodes, y0_nodes, f_nodes, df_ddy_nodes, bc1, bc2, j=1, rel=False, w=1,
                                               debug=False):
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
    :param rel: if True the residual is returned relative to f_nodes.
    :param w: the weight for Dy (default w=1.0).
    :param debug: if set to True outputs debug messages to stdout.
    :return: solution y = y0 + w * Dy; Dy; residual.
    """
    t0 = time.time()
    step = nodes[1:-1] - nodes[:-2]  # grid step
    m = fd_d2_matrix(nodes.size - 2) - sparse.diags([(j * step) ** 2 * df_ddy_nodes[1:-1]], [0],
                                                    (nodes.size - 2, nodes.size - 2), format='csc')
    dy = np.zeros_like(nodes)  # solution vector
    f = (j * step) ** 2 * f_nodes[1:-1] - fd_d2_matrix(nodes.size - 2).dot(y0_nodes[1:-1])
    f[0] -= bc1
    f[-1] -= bc2
    dy[0] = bc1 - y0_nodes[0]
    dy[-1] = bc2 - y0_nodes[-1]
    if debug:
        print('Time spent on matrix filling %2.2f s' % (time.time() - t0))
    t0 = time.time()
    if debug:
        print(m.todense())
    dy[1:-1] = linalg.spsolve(m, f, use_umfpack=True)
    y = y0_nodes + w * dy
    if debug:
        print('Time spent on solution %2.2f s' % (time.time() - t0))
    d_y0 = np.gradient(y0_nodes, nodes, edge_order=2) / j
    d2_y0 = np.gradient(d_y0, nodes, edge_order=2) / j
    d_dy = np.gradient(dy, nodes, edge_order=2) / j
    d2_dy = np.gradient(d_dy, nodes, edge_order=2) / j
    residual = f_nodes - d2_dy - d2_y0
    if rel:
        try:
            residual /= np.max(abs(f_nodes))
        except FloatingPointError:
            raise FloatingPointError()
    return y, dy, residual


def dirichlet_non_linear_poisson_solver(nodes, y0, f, df_ddy, bc1, bc2, j=1, rel=False, w=1, debug=False):
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
    :param rel: if True the residual is returned relative to f_nodes.
    :param w: the weight for Dy (default w=1.0).
    :param debug: if set to True outputs debug messages to stdout.
    :return: solution as callable function y = y0 + w * Dy; Dy; residual.
    """
    y0_nodes = y0(nodes)
    f_nodes = f(nodes, y0)
    df_ddy_nodes = df_ddy(nodes, y0)
    y_nodes, dy, residual = dirichlet_non_linear_poisson_solver_arrays(nodes, y0_nodes, f_nodes, df_ddy_nodes, bc1, bc2,
                                                                       j, rel, w, debug)
    y = interp_fn(nodes, y_nodes)
    return y, dy, residual


def dirichlet_non_linear_poisson_solver_mesh_arrays(mesh, y0_nodes, f_nodes, df_ddy_nodes, rel=False, w=1,
                                                    debug=False):
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
    :param rel: if True the residual is returned relative to f_nodes.
    :param w: the weight for Dy (default w=1.0)
    :param debug: if set to True outputs debug messages to stdout
    :return: mesh with solution y = y0 + w * Dy, and residual; Dy.
    """
    y_nodes, dy, residual = dirichlet_non_linear_poisson_solver_arrays(mesh.local_nodes, y0_nodes, f_nodes,
                                                                       df_ddy_nodes,
                                                                       mesh.boundary_condition_1,
                                                                       mesh.boundary_condition_2,
                                                                       mesh.jacobian, rel, w, debug)
    mesh.solution = y_nodes
    mesh.residual = residual
    return mesh, dy


def dirichlet_non_linear_poisson_solver_mesh(mesh, y0, f, df_ddy, rel=False, w=1, debug=False):
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
    :param rel: if True the residual is returned relative to f_nodes.
    :param w: the weight for Dy (default w=1.0)
    :param debug: if set to True outputs debug messages to stdout
    :return: mesh with solution y = y0 + w * Dy, and residual; callable solution function; Dy.
    """
    y0_nodes = y0(mesh.physical_nodes)
    f_nodes = f(mesh.physical_nodes, y0)
    df_ddy_nodes = df_ddy(mesh.physical_nodes, y0)
    mesh, dy = dirichlet_non_linear_poisson_solver_mesh_arrays(mesh, y0_nodes, f_nodes, df_ddy_nodes,
                                                               rel, w, debug)
    y = interp_fn(mesh.physical_nodes, mesh.solution)
    return mesh, y, dy


def dirichlet_non_linear_poisson_solver_recurrent_mesh(mesh, y0, f, df_ddy, max_iter=1000, threshold=1e-7, debug=False):
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
    :param debug: if set to True outputs debug messages to stdout
    :return: mesh with solution y = y0 + w * Dy, and residual; callable solution function.
    """
    for i in range(max_iter):
        if debug:
            print('Iteration:', i + 1)
        mesh, y0, dy = dirichlet_non_linear_poisson_solver_mesh(mesh, y0, f, df_ddy, debug=False)
        if debug:
            print('Integrated residual:', mesh.integrational_residual)
        if abs(mesh.integrational_residual) <= threshold or np.max(np.abs(dy)) <= 2 * np.finfo(np.float).eps:
            break
    return mesh, y0


def dirichlet_non_linear_poisson_solver_amr(boundary_1, boundary_2, step, y0, f, df_ddy, bc1, bc2,
                                            max_iter=1000, residual_threshold=1e-3, int_residual_threshold=1e-6,
                                            max_level=20, mesh_refinement_threshold=1e-7, debug=False):
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
        :param debug: if set to True outputs debug messages to stdout
        :return: meshes tree with solution and residual.
        """
    root_mesh = Mesh1DUniform(boundary_1, boundary_2,
                              boundary_condition_1=bc1,
                              boundary_condition_2=bc2,
                              physical_step=round(step, 9))
    meshes_tree = TreeMesh1DUniform(root_mesh, refinement_coefficient=2, aligned=True)
    while True:
        level = meshes_tree.levels[-1]
        converged = np.zeros(len(meshes_tree.tree[level]))
        refinements = []
        for mesh_id, mesh in enumerate(meshes_tree.tree[level]):
            mesh, y0 = dirichlet_non_linear_poisson_solver_recurrent_mesh(mesh, y0, f, df_ddy,
                                                                          max_iter, int_residual_threshold,
                                                                          debug=debug)
            mesh.trim()
            converged[mesh_id] = max(abs(mesh.residual)) < residual_threshold
            if converged[mesh_id]:
                continue
            refinement_points_chunks = points_for_refinement(mesh, mesh_refinement_threshold)
            converged[mesh_id] = np.all(np.array([block.size == 0 for block in refinement_points_chunks]))
            if converged[mesh_id]:
                continue
            elif level < max_level:
                for block in refinement_points_chunks:
                    idx1, idx2, mesh_crop = adjust_range(block, mesh.num - 1, crop=[10, 10],
                                                         step_scale=meshes_tree.refinement_coefficient)
                    refinements.append(Mesh1DUniform(
                        mesh.to_physical_coordinate(mesh.local_nodes[idx1]),
                        mesh.to_physical_coordinate(mesh.local_nodes[idx2]),
                        boundary_condition_1=mesh.solution[idx1],
                        boundary_condition_2=mesh.solution[idx2],
                        physical_step=mesh.physical_step/meshes_tree.refinement_coefficient,
                        crop=mesh_crop))
        meshes_tree.remove_coarse_duplicates()
        if converged.all() or level == max_level:
            break
        for refinement_mesh in refinements:
            meshes_tree.add_mesh(refinement_mesh)
        #flat_grid = meshes_tree.flatten()
        #y0 = interp1d(flat_grid.physical_nodes, y0(flat_grid.physical_nodes), kind='cubic')
    if debug:
        print('Mesh tree has ', meshes_tree.levels[-1], 'refinement levels')
    return meshes_tree
