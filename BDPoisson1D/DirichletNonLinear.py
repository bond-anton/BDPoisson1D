from __future__ import division, print_function

import time

import numpy as np
from scipy.interpolate import interp1d
from scipy import sparse
from scipy.sparse import linalg
from matplotlib import pyplot as plt

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
    :param w: the weight for Dy (default w=1.0)
    :param debug: if set to True outputs debug messages to stdout
    :return: solution y = y0 + w * Dy; Dy; residual
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
    :param w: the weight for Dy (default w=1.0)
    :param debug: if set to True outputs debug messages to stdout
    :return: solution as callable function y = y0 + w * Dy; Dy; residual
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
    :return: mesh with solution y = y0 + w * Dy, and residual; Dy;
    """
    y_nodes, dy, residual = dirichlet_non_linear_poisson_solver_arrays(mesh.local_nodes, y0_nodes, f_nodes,
                                                                       df_ddy_nodes,
                                                                       mesh.bc1, mesh.bc2, mesh.j, rel, w, debug)
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
    :return: mesh with solution y = y0 + w * Dy, and residual; Dy;
    """
    y0_nodes = y0(mesh.phys_nodes)
    f_nodes = f(mesh.phys_nodes, y0)
    df_ddy_nodes = df_ddy(mesh.phys_nodes, y0)
    mesh, dy = dirichlet_non_linear_poisson_solver_mesh_arrays(mesh, y0_nodes, f_nodes, df_ddy_nodes,
                                                               rel, w, debug)
    y = interp_fn(mesh.phys_nodes, mesh.solution)
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
    :return: mesh with solution y = y0 + w * Dy, and residual; Dy;
    """
    for i in range(max_iter):
        if debug:
            print('Iteration:', i + 1)
        mesh, y0, dy = dirichlet_non_linear_poisson_solver_mesh(mesh, y0, f, df_ddy, debug=False)
        if debug:
            print('Integrated residual:', mesh.int_residual)
        if abs(mesh.integrational_residual) <= threshold or np.max(abs(dy)) <= 2 * np.finfo(np.float).eps:
            break
    return mesh, y0


def dirichlet_non_linear_poisson_solver_amr(nodes, Psi, f, dfdDPsi, bc1, bc2,
                                            max_iterations=1000, residual_threshold=1e-3, int_residual_threshold=1e-6,
                                            max_level=20, mesh_refinement_threshold=1e-7, debug=False):
    '''
    The recurrent NL Poisson solver with the Adaptive Mesh Refinement
    nodes is the initial physical mesh
    '''
    root_mesh = MeshUniform1D(nodes[0], nodes[-1], nodes[1] - nodes[0], bc1, bc2)
    Meshes = TreeMeshUniform1D(root_mesh, refinement_coefficient=2, aligned=True)
    converged = np.zeros(1)
    level = 0
    while (not converged.all() or level < Meshes.levels[-1]) and level <= max_level:
        if debug:
            print('Solving for Meshes of level:', level, 'of', Meshes.levels[-1])
        converged = np.zeros(len(Meshes.tree[level]))
        for mesh_id, mesh in enumerate(Meshes.tree[level]):
            mesh, Psi = dirichlet_non_linear_poisson_solver_reccurent_mesh(mesh, Psi, f, dfdDPsi,
                                                                           max_iterations, int_residual_threshold,
                                                                           debug=debug)
            mesh.trim()
            if max(abs(mesh.residual)) < residual_threshold:
                if debug:
                    print('CONVERGED!')
                converged[mesh_id] = True
                continue
            refinement_points_chunks = points_for_refinement(mesh, mesh_refinement_threshold)
            if len(refinement_points_chunks) == 0 or np.all(
                    np.array([block.size == 0 for block in refinement_points_chunks])):
                if debug:
                    print('CONVERGED!')
                converged[mesh_id] = True
                continue
            if level < max_level:
                if debug:
                    print('nodes for refinement:', refinement_points_chunks)
                for block in refinement_points_chunks:
                    idx1, idx2, crop = adjust_range(block, mesh.num - 1, crop=[3, 3], step_scale=2)
                    start_point = mesh.to_physical_coordinate(mesh.local_nodes[idx1])
                    stop_point = mesh.to_physical_coordinate(mesh.local_nodes[idx2])
                    ref_bc1 = mesh.solution[idx1]
                    ref_bc2 = mesh.solution[idx2]
                    print(start_point, stop_point)
                    print(ref_bc1, ref_bc2)
                    refinement_mesh = MeshUniform1D(start_point, stop_point,
                                                    mesh.physical_step / Meshes.refinement_coefficient,
                                                    ref_bc1, ref_bc2,
                                                    crop=crop)
                    # print 'CROP:', crop
                    Meshes.add_mesh(refinement_mesh)
                    flat_grid, _, _ = Meshes.flatten()
                    Psi = interp1d(flat_grid, Psi(flat_grid), kind='cubic')
                    if debug:
                        _, ax = plt.subplots(1)
                        ax.plot(mesh.physical_nodes, mesh.solution, 'b-o')
                        ax.plot(refinement_mesh.physical_nodes, Psi(refinement_mesh.physical_nodes), 'r-o')
                        plt.show()
                if debug:
                    Meshes.plot_tree()
        level += 1
    if debug: print('Mesh tree has ', Meshes.levels[-1], 'refinement levels')
    return Meshes
