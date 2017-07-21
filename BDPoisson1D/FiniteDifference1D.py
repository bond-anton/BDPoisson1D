from __future__ import division, print_function

import time

import numpy as np
from scipy.interpolate import interp1d
from scipy import sparse
from scipy.sparse import linalg
from matplotlib import pyplot as plt

from BDMesh import UniformMesh1D, Uniform1DMeshesTree
from ._helpers import interp_Fn


def fd_d2_matrix(size):
    """
    d2 finite difference matrix generator
    :param size: size of matrix to generate (int)
    :return: d2 finite difference sparse matrix of three diagonals of O(h2) precision.
    """
    a = -2 * np.ones(size)
    b = np.ones(size - 1)
    c = np.ones(size - 1)
    return sparse.diags([c, a, b], offsets=[-1, 0, 1], shape=(size, size), format='csc')


def dirichlet_poisson_solver_arrays(nodes, f_nodes, bc1, bc2, j=1):
    """
    Solves 1D differential equation of the form
        d2y/dx2 = f(x)
        y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
    using FDE algorithm of O(h2) precision.

    :param nodes: 1D array of x nodes. Must include boundary points.
    :param f_nodes: 1D array of values of f(x) on nodes array. Must be same shape as nodes.
    :param bc1: boundary condition at nodes[0] point (a number).
    :param bc2: boundary condition at nodes[0] point (a number).
    :param j: Jacobian.
    :return:
        y: 1D array of solution function y(x) values on nodes array.
        residual: error of the solution.
    """
    step = nodes[1:-1] - nodes[:-2]  # grid step
    m = fd_d2_matrix(nodes.size - 2)
    y = np.array([bc1] + [0] * (nodes.size - 2) + [bc2])  # solution vector
    f = (j * step) ** 2 * f_nodes[1:-1] - np.delete(y, [1, 2])
    y[1:-1] = linalg.spsolve(m, f, use_umfpack=True)
    dy = np.gradient(y, nodes, edge_order=2) / j
    d2y = np.gradient(dy, nodes, edge_order=2) / j
    residual = f_nodes - d2y
    return y, residual


def dirichlet_poisson_solver(nodes, f, bc1, bc2, j=1):
    """
    Solves 1D differential equation of the form
        d2y/dx2 = f(x)
        y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
    using FDE algorithm of O(h2) precision.

    :param nodes: 1D array of x nodes. Must include boundary points.
    :param f: function f(x) callable on nodes array.
    :param bc1: boundary condition at nodes[0] point (a number).
    :param bc2: boundary condition at nodes[0] point (a number).
    :param j: Jacobian.
    :return:
        y: 1D array of solution function y(x) values on nodes array.
        residual: error of the solution.
    """
    return dirichlet_poisson_solver_arrays(nodes, f(nodes), bc1, bc2, j)


def dirichlet_poisson_solver_mesh_arrays(mesh, f_nodes):
    """
    Solves 1D differential equation of the form
        d2y/dx2 = f(x)
        y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
    using FDE algorithm of O(h2) precision

    :param mesh: BDMesh to solve on.
    :param f_nodes: 1D array of values of f(x) on nodes array. Must be same shape as nodes.
    :return: mesh with solution and residual
    """
    y, residual = dirichlet_poisson_solver_arrays(mesh.local_nodes, f_nodes, mesh.bc1, mesh.bc2, mesh.J)
    mesh.solution = y
    mesh.residual = residual
    mesh.int_residual = np.trapz(mesh.residual, mesh.phys_nodes())
    return mesh


def dirichlet_poisson_solver_mesh(mesh, f):
    """
    Solves 1D differential equation of the form
        d2y/dx2 = f(x)
        y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
    using FDE algorithm of O(h2) precision

    :param mesh: BDMesh to solve on.
    :param f: function f(x) callable on nodes array.
    :return: mesh with solution and residual
    """
    return dirichlet_poisson_solver_mesh_arrays(mesh, f(mesh.phys_nodes()))


def neumann_poisson_solver_arrays(nodes, f_nodes, bc1, bc2, j=1, y0=0, debug=False):
    """
    Solves 1D differential equation of the form
        d2y/dx2 = f(x)
        dy/dx(x0) = bc1, dy/dx(xn) = bc2 (Neumann boundary condition)
    using FDE algorithm of O(h2) precision.

    :param nodes: 1D array of x nodes. Must include boundary points.
    :param f_nodes: 1D array of values of f(x) on nodes array. Must be same shape as nodes.
    :param bc1: boundary condition at nodes[0] point (a number).
    :param bc2: boundary condition at nodes[0] point (a number).
    :param j: Jacobian.
    :param y0: value of y(x) at point x0 of nodes array
    :return:
        y: 1D array of solution function y(x) values on nodes array.
        residual: error of the solution.
    """
    integral = np.trapz(f_nodes, nodes)
    if not np.allclose(integral, bc2 - bc1):
        print('WARNING!!!!')
        print('The problem is not well-posed!')
        print('Redefine the f function and BCs or refine the mesh!')
        print('WARNING!!!!')
    step = nodes[1:] - nodes[:-1]  # grid step
    m = fd_d2_matrix(nodes.size - 1)
    m[0, 0] = 0
    m[-1, -2] = 2
    y = np.append([y0], np.zeros(nodes.size - 1))  # solution vector
    f = (j * step) ** 2 * f_nodes[1:]
    f[0] += step[0] ** 2 * f_nodes[0] + 2 * step[0] * bc1 + y0
    f[-1] -= 2 * step[-1] * bc2
    y[1:] = linalg.spsolve(m, f, use_umfpack=True)
    dy = np.gradient(y, nodes, edge_order=2) / j
    d2y = np.gradient(dy, nodes, edge_order=2) / j
    residual = f_nodes - d2y
    return y, residual


def neumann_poisson_solver(nodes, f, bc1, bc2, j=1, y0=0):
    """
    Solves 1D differential equation of the form
        d2y/dx2 = f(x)
        dy/dx(x0) = bc1, dy/dx(xn) = bc2 (Neumann boundary condition)
    using FDE algorithm of O(h2) precision.

    :param nodes: 1D array of x nodes. Must include boundary points.
    :param f: function f(x) callable on nodes array..
    :param bc1: boundary condition at nodes[0] point (a number).
    :param bc2: boundary condition at nodes[0] point (a number).
    :param j: Jacobian.
    :param y0: value of y(x) at point x0 of nodes array
    :return:
        y: 1D array of solution function y(x) values on nodes array.
        residual: error of the solution.
    """
    return neumann_poisson_solver_arrays(nodes, f(nodes), bc1, bc2, j, y0)


def dirichlet_non_linear_poisson_solver_arrays(nodes, Psi0_nodes, f_nodes, dfdDPsi_nodes, bc1, bc2, J=1, rel=False, W=1,
                                               debug=False):
    '''
    solves equation of form d2Psi/dx2 = f(Psi(x))
    Psi(x0) = bc1, Psi(xn) = bc2
    using FDE algorithm of O(h2) precision
    and Tailor series for linearization.
    '''
    t0 = time.time()
    h = nodes[1:-1] - nodes[:-2]  # grid step
    M = fd_d2_matrix(nodes.size - 2) - sparse.diags([(J * h) ** 2 * dfdDPsi_nodes[1:-1]], [0],
                                                    (nodes.size - 2, nodes.size - 2), format='csc')
    DPsi = np.zeros_like(nodes)  # solution vector
    F = (J * h) ** 2 * f_nodes[1:-1] - fd_d2_matrix(nodes.size - 2).dot(Psi0_nodes[1:-1])
    F[0] -= bc1
    F[-1] -= bc2
    DPsi[0] = bc1 - Psi0_nodes[0]
    DPsi[-1] = bc2 - Psi0_nodes[-1]
    if debug: print
    'Time spent on matrix filling %2.2f s' % (time.time() - t0)
    t0 = time.time()
    if debug: print
    M.todense()
    DPsi[1:-1] = linalg.spsolve(M, F, use_umfpack=True)
    Psi = Psi0_nodes + W * DPsi
    if debug: print
    'Time spent on solution %2.2f s' % (time.time() - t0)
    dx = np.gradient(nodes)
    dPsi0 = np.gradient(Psi0_nodes, dx, edge_order=2) / J
    d2Psi0 = np.gradient(dPsi0, dx, edge_order=2) / J
    # dPsi = np.gradient(Psi(nodes), dx, edge_order=2) / J
    # d2Psi = np.gradient(dPsi, dx, edge_order=2) / J
    dDPsi = np.gradient(DPsi, dx, edge_order=2) / J
    d2DPsi = np.gradient(dDPsi, dx, edge_order=2) / J
    # print f_nodes
    R = f_nodes - d2DPsi - d2Psi0
    # print R
    if rel:
        try:
            R /= np.max(abs(f_nodes))
        except FloatingPointError:
            raise FloatingPointError()
    return Psi, DPsi, R


def dirichlet_non_linear_poisson_solver(nodes, Psi0, f, dfdDPsi, bc1, bc2, J=1, rel=False, W=1, debug=False):
    '''
    solves equation of form d2Psi/dx2 = f(Psi(x))
    Psi(x0) = bc1, Psi(xn) = bc2
    using FDE algorithm of O(h2) precision
    and Tailor series for linearization.
    '''
    Psi0_nodes = Psi0(nodes)
    f_nodes = f(nodes, Psi0)
    dfdDPsi_nodes = dfdDPsi(nodes, Psi0)
    Psi_nodes, DPsi, R = dirichlet_non_linear_poisson_solver_arrays(nodes, Psi0_nodes, f_nodes, dfdDPsi_nodes, bc1, bc2,
                                                                    J, rel, W, debug)
    Psi = interp_Fn(nodes, Psi_nodes)
    return Psi, DPsi, R


def dirichlet_non_linear_poisson_solver_mesh_arrays(mesh, Psi0_nodes, f_nodes, dfdDPsi_nodes, rel=False, W=1,
                                                    debug=False):
    '''
    solves equation of form d2Psi/dx2 = f(Psi(x))
    Psi(x0) = bc1, Psi(xn) = bc2
    using FDE algorithm of O(h2) precision
    and Tailor series for linearization.
    '''
    Psi_nodes, DPsi, R = dirichlet_non_linear_poisson_solver_arrays(mesh.local_nodes, Psi0_nodes, f_nodes,
                                                                    dfdDPsi_nodes,
                                                                    mesh.bc1, mesh.bc2, mesh.J, rel, W, debug)
    # Psi_nodes, DPsi, R = dirichlet_non_linear_poisson_solver_arrays(mesh.phys_nodes(), Psi0_nodes, f_nodes, dfdDPsi_nodes,
    #                                                                mesh.bc1, mesh.bc2, 1, debug)
    mesh.solution = Psi_nodes
    mesh.residual = R
    mesh.int_residual = np.trapz(mesh.residual, mesh.phys_nodes())
    return mesh, Psi_nodes, DPsi


def dirichlet_non_linear_poisson_solver_mesh(mesh, Psi0, f, dfdDPsi, rel=False, W=1, debug=False):
    '''
    solves equation of form d2Psi/dx2 = f(Psi(x))
    Psi(x0) = bc1, Psi(xn) = bc2
    using FDE algorithm of O(h2) precision
    and Tailor series for linearization.
    '''
    Psi0_nodes = Psi0(mesh.phys_nodes())
    f_nodes = f(mesh.phys_nodes(), Psi0)
    dfdDPsi_nodes = dfdDPsi(mesh.phys_nodes(), Psi0)
    mesh, Psi_nodes, DPsi = dirichlet_non_linear_poisson_solver_mesh_arrays(mesh, Psi0_nodes, f_nodes, dfdDPsi_nodes,
                                                                            rel, W, debug)
    Psi = interp_Fn(mesh.phys_nodes(), Psi_nodes)  # , interp_type='last')
    return mesh, Psi, DPsi


def dirichlet_non_linear_poisson_solver_reccurent_mesh(mesh, Psi0, f, dfdDPsi,
                                                       max_iterations=1000, threshold=1e-7,
                                                       debug=False):
    '''
    solves equation of form d2Psi/dx2 = f(Psi(x))
    Psi(x0) = bc1, Psi(xn) = bc2
    using FDE algorithm of O(h2) precision
    and Tailor series for linearization.
    '''
    mesh.int_residual = threshold + 1
    int_residual_array = []
    dx = np.gradient(mesh.phys_nodes())
    if debug:
        DPsi = np.ones_like(mesh.local_nodes)
        E = np.zeros_like(mesh.local_nodes)
        dPsi = np.gradient(Psi0(mesh.phys_nodes()), dx, edge_order=2)
        d2Psi = np.gradient(dPsi, dx, edge_order=2)

        plt.ion()
        _, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
        ax1.set_autoscaley_on(True)
        ax2.set_autoscaley_on(True)
        ax3.set_autoscaley_on(True)
        ax4.set_autoscaley_on(True)
        ax4.grid(1)
        ax5.set_autoscalex_on(True)
        ax5.set_autoscaley_on(True)

        Psi_line, = ax1.plot(mesh.local_nodes, Psi0(mesh.phys_nodes()))
        DPsi_line, = ax2.plot(mesh.local_nodes, DPsi)
        f_line, = ax3.plot(mesh.local_nodes, f(mesh.phys_nodes(), Psi0))
        d2Psi_line, = ax3.plot(mesh.local_nodes, d2Psi)
        E_line, = ax4.plot(mesh.local_nodes, E)
        R_line, = ax5.plot(int_residual_array)
        plt.draw()

    i = 1
    while abs(mesh.int_residual) > threshold and i < max_iterations and np.max(abs(DPsi)) > 2 * np.finfo(np.float).eps:
        if debug:
            print('Iteration:', i)
        # time.sleep(1)
        mesh, Psi0, DPsi = dirichlet_non_linear_poisson_solver_mesh(mesh, Psi0, f, dfdDPsi, debug=False)
        int_residual_array.append(mesh.int_residual)
        if debug:
            print('Integrated residual:', mesh.int_residual)
        if debug:
            Psi_line.set_ydata(mesh.solution)
            f_line.set_ydata(f(mesh.phys_nodes(), Psi0))
            dPsi = np.gradient(mesh.solution, dx, edge_order=2)
            d2Psi = np.gradient(dPsi, dx, edge_order=2)
            DPsi_line.set_ydata(DPsi)
            d2Psi_line.set_ydata(d2Psi)
            E_line.set_ydata(mesh.residual)
            # E_line.set_ydata(dfdDPsi(mesh.phys_nodes(), Psi0))
            R_line.set_data(np.arange(i), int_residual_array)
            # R_line.set_ydata(int_residual_array)
            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()
            ax3.relim()
            ax3.autoscale_view()
            ax4.relim()
            ax4.autoscale_view()
            ax5.relim()
            ax5.autoscale_view()
            plt.draw()
        i += 1

    if debug: plt.ioff()
    return mesh, Psi0


def dirichlet_poisson_solver_amr(nodes, f, bc1, bc2, threshold, max_level=20):
    '''
    The same as above but uses the Adaptive Mesh Refinement
    nodes is the initial physical mesh
    '''
    root_mesh = UniformMesh1D(nodes[0], nodes[-1], nodes[1] - nodes[0], bc1, bc2)
    Meshes = Uniform1DMeshesTree(root_mesh, refinement_coefficient=2, aligned=True)
    converged = np.zeros(1)
    level = 0
    while (not converged.all() or level < Meshes.levels[-1]) and level <= max_level:
        print('Solving for Meshes of level:', level, 'of', Meshes.levels[-1])
        converged = np.zeros(len(Meshes.Tree[level]))
        for mesh_id, mesh in enumerate(Meshes.Tree[level]):
            mesh = dirichlet_poisson_solver_mesh(mesh, f, debug=False)
            mesh.trim()
            refinement_points_chunks = points_for_refinement(mesh, threshold)
            if len(refinement_points_chunks) == 0 or np.all(
                    np.array([block.size == 0 for block in refinement_points_chunks])):
                print('CONVERGED!')
                converged[mesh_id] = True
                continue
            if level < max_level:
                print('nodes for refinement:', refinement_points_chunks)
                for block in refinement_points_chunks:
                    idx1, idx2, mesh_crop = adjust_range(block, mesh.num - 1, crop=[3, 3],
                                                         step_scale=Meshes.refinement_coefficient)
                    start_point = mesh.to_phys(mesh.local_nodes[idx1])
                    stop_point = mesh.to_phys(mesh.local_nodes[idx2])
                    ref_bc1 = mesh.solution[idx1]
                    ref_bc2 = mesh.solution[idx2]
                    refinement_mesh = UniformMesh1D(start_point, stop_point,
                                                    mesh.phys_step / Meshes.refinement_coefficient, ref_bc1, ref_bc2,
                                                    crop=mesh_crop)
                    # print 'CROP:', crop
                    Meshes.add_mesh(refinement_mesh)
                    # Meshes.plot_tree()
        level += 1
        print()
    print('Mesh tree has ', Meshes.levels[-1], 'refinement levels')
    return Meshes


def dirichlet_non_linear_poisson_solver_amr(nodes, Psi, f, dfdDPsi, bc1, bc2,
                                            max_iterations=1000, residual_threshold=1e-3, int_residual_threshold=1e-6,
                                            max_level=20, mesh_refinement_threshold=1e-7, debug=False):
    '''
    The recurrent NL Poisson solver with the Adaptive Mesh Refinement
    nodes is the initial physical mesh
    '''
    root_mesh = UniformMesh1D(nodes[0], nodes[-1], nodes[1] - nodes[0], bc1, bc2)
    Meshes = Uniform1DMeshesTree(root_mesh, refinement_coefficient=2, aligned=True)
    converged = np.zeros(1)
    level = 0
    while (not converged.all() or level < Meshes.levels[-1]) and level <= max_level:
        if debug:
            print('Solving for Meshes of level:', level, 'of', Meshes.levels[-1])
        converged = np.zeros(len(Meshes.Tree[level]))
        for mesh_id, mesh in enumerate(Meshes.Tree[level]):
            mesh, Psi = dirichlet_non_linear_poisson_solver_reccurent_mesh(mesh, Psi, f, dfdDPsi,
                                                                           max_iterations, int_residual_threshold,
                                                                           debug=debug)
            mesh.trim()
            if max(abs(mesh.residual)) < residual_threshold:
                if debug: print
                'CONVERGED!'
                converged[mesh_id] = True
                continue
            refinement_points_chunks = points_for_refinement(mesh, mesh_refinement_threshold)
            if len(refinement_points_chunks) == 0 or np.all(
                    np.array([block.size == 0 for block in refinement_points_chunks])):
                if debug: print
                'CONVERGED!'
                converged[mesh_id] = True
                continue
            if level < max_level:
                if debug:
                    print('nodes for refinement:', refinement_points_chunks)
                for block in refinement_points_chunks:
                    idx1, idx2, crop = adjust_range(block, mesh.num - 1, crop=[3, 3], step_scale=2)
                    start_point = mesh.to_phys(mesh.local_nodes[idx1])
                    stop_point = mesh.to_phys(mesh.local_nodes[idx2])
                    ref_bc1 = mesh.solution[idx1]
                    ref_bc2 = mesh.solution[idx2]
                    print(start_point, stop_point)
                    print(ref_bc1, ref_bc2)
                    refinement_mesh = UniformMesh1D(start_point, stop_point,
                                                    mesh.phys_step / Meshes.refinement_coefficient, ref_bc1, ref_bc2,
                                                    crop=crop)
                    # print 'CROP:', crop
                    Meshes.add_mesh(refinement_mesh)
                    flat_grid, _, _ = Meshes.flatten()
                    Psi = interp1d(flat_grid, Psi(flat_grid), kind='cubic')
                    if debug:
                        _, ax = plt.subplots(1)
                        ax.plot(mesh.phys_nodes(), mesh.solution, 'b-o')
                        ax.plot(refinement_mesh.phys_nodes(), Psi(refinement_mesh.phys_nodes()), 'radius-o')
                        plt.show()
                if debug: Meshes.plot_tree()
        level += 1
    if debug: print('Mesh tree has ', Meshes.levels[-1], 'refinement levels')
    return Meshes


def adjust_range(idx_range, max_index, crop=[0, 0], step_scale=1):
    idx1 = idx_range[0] if idx_range[0] >= 0 else 0
    idx2 = idx_range[-1] if idx_range[-1] <= max_index else max_index
    mesh_crop = [0, 0]
    if idx2 - idx1 < 2:
        if idx1 == 0 and idx2 != max_index:
            idx2 += 1
        elif idx2 == max_index and idx1 != 0:
            idx1 -= 1
        elif idx2 == max_index and idx1 == 0:
            raise Exception('the range is too short!')
        else:
            idx1 -= 1
            idx2 += 1
    if (idx1 - np.floor(step_scale * crop[0])) >= 0:
        mesh_crop[0] = int(crop[0])
    else:
        # print 'idx1'
        mesh_crop[0] = int(np.floor(idx1 / step_scale))
    idx1 -= int(np.floor(step_scale * mesh_crop[0]))
    if (idx2 + np.ceil(step_scale * crop[1])) <= max_index:
        mesh_crop[1] = int(crop[1])
    else:
        # print 'idx2'
        mesh_crop[1] = int(np.floor((max_index - idx2) / step_scale))
        # print mesh_crop[1], int(np.floor(step_scale*mesh_crop[1]))
    idx2 += int(np.floor(step_scale * mesh_crop[1]))
    return idx1, idx2, mesh_crop


def points_for_refinement(mesh, threshold):
    bad_nodes = np.sort(np.where(abs(mesh.residual) > threshold)[0])
    split_idx = np.where(bad_nodes[1:] - bad_nodes[:-1] > 1)[0] + 1
    bad_nodes = np.split(bad_nodes, split_idx)
    return bad_nodes