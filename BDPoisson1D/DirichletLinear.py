from __future__ import division, print_function

import numpy as np
from scipy.sparse import linalg

from BDMesh import Mesh1DUniform, TreeMesh1DUniform
from ._helpers import fd_d2_matrix, points_for_refinement, adjust_range


def dirichlet_poisson_solver_arrays(nodes, f_nodes, bc1, bc2, j=1):
    """
    Solves 1D differential equation of the form
        d2y/dx2 = f(x)
        y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
    using FDE algorithm of O(h2) precision.

    :param nodes: 1D array of x nodes. Must include boundary points.
    :param f_nodes: 1D array of values of f(x) on nodes array. Must be same shape as nodes.
    :param bc1: boundary condition at nodes[0] point (a number).
    :param bc2: boundary condition at nodes[n] point (a number).
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
    :param bc2: boundary condition at nodes[n] point (a number).
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
    using FDE algorithm of O(h2) precision.

    :param mesh: BDMesh to solve on.
    :param f_nodes: 1D array of values of f(x) on nodes array. Must be same shape as nodes.
    :return: mesh with solution and residual.
    """
    assert isinstance(mesh, Mesh1DUniform)
    y, residual = dirichlet_poisson_solver_arrays(mesh.local_nodes, f_nodes,
                                                  mesh.boundary_condition_1, mesh.boundary_condition_2,
                                                  mesh.jacobian)
    mesh.solution = y
    mesh.residual = residual
    return mesh


def dirichlet_poisson_solver_mesh(mesh, f):
    """
    Solves 1D differential equation of the form
        d2y/dx2 = f(x)
        y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
    using FDE algorithm of O(h2) precision.

    :param mesh: BDMesh to solve on.
    :param f: function f(x) callable on nodes array.
    :return: mesh with solution and residual.
    """
    assert isinstance(mesh, Mesh1DUniform)
    return dirichlet_poisson_solver_mesh_arrays(mesh, f(mesh.physical_nodes))


def dirichlet_poisson_solver_amr(boundary_1, boundary_2, step, f, bc1, bc2,
                                 threshold=1e-2, max_level=10, verbose=False):
    """
    Linear Poisson equation solver with Adaptive Mesh Refinement algorithm.
    :param boundary_1: physical nodes left boundary.
    :param boundary_2: physical nodes right boundary.
    :param step: physical nodes step.
    :param f: function f(x) callable on nodes array.
    :param bc1: boundary condition at nodes[0] point (a number).
    :param bc2: boundary condition at nodes[n] point (a number).
    :param threshold: algorithm convergence residual threshold value.
    :param max_level: max level of mesh refinement.
    :return: meshes tree with solution and residual.
    """
    root_mesh = Mesh1DUniform(boundary_1, boundary_2,
                              boundary_condition_1=bc1,
                              boundary_condition_2=bc2,
                              physical_step=round(step, 9))
    meshes_tree = TreeMesh1DUniform(root_mesh, refinement_coefficient=2, aligned=False)
    converged = np.zeros(1)
    level = meshes_tree.levels[-1]
    while level < max_level:
        level = meshes_tree.levels[-1]
        if converged.all():
            break
        if verbose:
            print('\nSolving for', len(meshes_tree.tree[level]),'Meshes of level:', level, 'of', meshes_tree.levels[-1])
        converged = np.zeros(len(meshes_tree.tree[level]))
        refinements = []
        for mesh_id, mesh in enumerate(meshes_tree.tree[level]):
            mesh = dirichlet_poisson_solver_mesh(mesh, f)
            mesh.trim()
            refinement_points_chunks = points_for_refinement(mesh, threshold)
            if np.all(np.array([block.size == 0 for block in refinement_points_chunks])):
                if verbose:
                    print('=====> CONVERGED!')
                converged[mesh_id] = True
                continue
            if level < max_level:
                for block in refinement_points_chunks:
                    idx1, idx2, mesh_crop = adjust_range(block, mesh.num - 1, crop=[10, 10],
                                                         step_scale=meshes_tree.refinement_coefficient)
                    start_point = mesh.to_physical_coordinate(mesh.local_nodes[idx1])
                    stop_point = mesh.to_physical_coordinate(mesh.local_nodes[idx2])
                    ref_bc1 = mesh.solution[idx1]
                    ref_bc2 = mesh.solution[idx2]
                    refinement_mesh = Mesh1DUniform(start_point, stop_point,
                                                    boundary_condition_1=ref_bc1,
                                                    boundary_condition_2=ref_bc2,
                                                    physical_step=mesh.physical_step/meshes_tree.refinement_coefficient,
                                                    crop=mesh_crop)
                    refinements.append(refinement_mesh)
        meshes_tree.remove_coarse_duplicates()
        for refinement_mesh in refinements:
            meshes_tree.add_mesh(refinement_mesh)
    if verbose:
        print('Mesh tree has ', meshes_tree.levels[-1], 'refinement levels')
    return meshes_tree
