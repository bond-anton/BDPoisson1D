from __future__ import division, print_function

import numpy as np
from scipy.sparse import linalg

from BDMesh import Mesh1DUniform, TreeMesh1DUniform
from ._helpers cimport fd_d2_matrix, points_for_refinement, adjust_range


cpdef dirichlet_poisson_solver_arrays(double[:] nodes, double[:] f_nodes, double bc1, double bc2, double j=1.0):
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
    cdef:
        double step = nodes[1] - nodes[0]
        double[:] y, dy, d2y, residual, sol
        int i
    #step = np.array(nodes[1:-1]) - np.array(nodes[:-2])  # grid step
    m = fd_d2_matrix(nodes.size - 2)
    y = np.array([bc1] + [0] * (nodes.size - 2) + [bc2])  # solution vector
    f = (j * step) ** 2 * np.array(f_nodes[1:-1]) - np.delete(y, [1, 2])
    sol = linalg.spsolve(m, f, use_umfpack=True)
    for i in range(sol.size):
        y[i+1] = sol[i]
    print(np.array(y))
    dy = np.gradient(y, nodes, edge_order=2) / j
    d2y = np.gradient(dy, nodes, edge_order=2) / j
    residual = np.array(f_nodes) - np.array(d2y)
    return np.array(y), np.array(residual)


cpdef dirichlet_poisson_solver(double[:] nodes, f, double bc1, double bc2, double j=1.0):
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


cpdef dirichlet_poisson_solver_mesh_arrays(mesh, double[:] f_nodes):
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
    print(type(mesh.boundary_condition_1))
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
                                 threshold=1e-2, max_level=10):
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
    meshes_tree = TreeMesh1DUniform(root_mesh, refinement_coefficient=2, aligned=True)
    while True:
        level = meshes_tree.levels[-1]
        converged = np.zeros(len(meshes_tree.tree[level]))
        refinements = []
        for mesh_id, mesh in enumerate(meshes_tree.tree[level]):
            mesh = dirichlet_poisson_solver_mesh(mesh, f)
            mesh.trim()
            refinement_points_chunks = points_for_refinement(mesh, threshold)
            converged[mesh_id] = np.all(np.array([block.size == 0 for block in refinement_points_chunks]))
            if converged[mesh_id]:
                continue
            elif level < max_level:
                for block in refinement_points_chunks:
                    idx1, idx2, mesh_crop = adjust_range(block, mesh.num - 1, crop=[10, 10],
                                                         step_scale=meshes_tree.refinement_coefficient)
                    refinements.append(Mesh1DUniform(
                        mesh.to_physical_coordinate(np.array([mesh.local_nodes[idx1]]))[0],
                        mesh.to_physical_coordinate(np.array([mesh.local_nodes[idx2]]))[0],
                        boundary_condition_1=mesh.solution[idx1],
                        boundary_condition_2=mesh.solution[idx2],
                        physical_step=mesh.physical_step/meshes_tree.refinement_coefficient,
                        crop=mesh_crop))
        meshes_tree.remove_coarse_duplicates()
        if converged.all() or level == max_level:
            break
        for refinement_mesh in refinements:
            meshes_tree.add_mesh(refinement_mesh)
    return meshes_tree
