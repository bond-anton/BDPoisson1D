from __future__ import division, print_function

import numpy as np
from scipy.sparse import linalg

from cython cimport boundscheck, wraparound

from BDMesh.TreeMesh1DUniform cimport TreeMesh1DUniform
from BDMesh.Mesh1DUniform cimport Mesh1DUniform
from ._helpers cimport fd_d2_matrix, points_for_refinement, adjust_range
from .Function cimport Function


@boundscheck(False)
@wraparound(False)
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
        int i, n = nodes.size
        double[:] dy, d2y, sol
        double[:] y = np.zeros(n, dtype=np.double)
        double[:] residual = np.zeros(n, dtype=np.double)
        double[:] step = np.zeros(n - 2, dtype=np.double)
        double[:] f = np.zeros(n - 2, dtype=np.double)
    y[0] = bc1
    y[n - 1] = bc2
    m = fd_d2_matrix(n - 2)
    for i in range(n - 2):
        step[i] = nodes[i + 1] - nodes[i]  # grid step
        f[i] = (j * step[i]) ** 2 * f_nodes[i + 1]
    f[0] -= bc1
    f[n - 3] -= bc2
    sol = linalg.spsolve(m, f, use_umfpack=True)
    for i in range(n - 2):
        y[i + 1] = sol[i]
    dy = np.gradient(y, nodes, edge_order=2) / j
    d2y = np.gradient(dy, nodes, edge_order=2) / j
    for i in range(n):
        residual[i] = f_nodes[i] - d2y[i]
    return np.array(y), np.array(residual)


cpdef dirichlet_poisson_solver(double[:] nodes, Function f, double bc1, double bc2, double j=1.0):
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
    return dirichlet_poisson_solver_arrays(nodes, f.evaluate(nodes), bc1, bc2, j)


cpdef dirichlet_poisson_solver_mesh_arrays(Mesh1DUniform mesh, double[:] f_nodes):
    """
    Solves 1D differential equation of the form
        d2y/dx2 = f(x)
        y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
    using FDE algorithm of O(h2) precision.

    :param mesh: BDMesh to solve on.
    :param f_nodes: 1D array of values of f(x) on nodes array. Must be same shape as nodes.
    :return: mesh with solution and residual.
    """
    y, residual = dirichlet_poisson_solver_arrays(mesh.__local_nodes, f_nodes,
                                                  mesh.__boundary_condition_1, mesh.__boundary_condition_2,
                                                  mesh.j())
    mesh.solution = y
    mesh.residual = residual
    return mesh


cpdef dirichlet_poisson_solver_mesh(Mesh1DUniform mesh, Function f):
    """
    Solves 1D differential equation of the form
        d2y/dx2 = f(x)
        y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
    using FDE algorithm of O(h2) precision.

    :param mesh: BDMesh to solve on.
    :param f: function f(x) callable on nodes array.
    :return: mesh with solution and residual.
    """
    return dirichlet_poisson_solver_mesh_arrays(mesh, f.evaluate(mesh.to_physical(mesh.__local_nodes)))


@boundscheck(False)
@wraparound(False)
cpdef dirichlet_poisson_solver_amr(double boundary_1, double boundary_2, double step, Function f,
                                   double bc1, double bc2,
                                   double threshold=1e-2, int max_level=10):
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
    cdef:
        Mesh1DUniform root_mesh, mesh
        TreeMesh1DUniform meshes_tree
        int level, mesh_id, idx1, idx2
        list refinements, refinement_points_chunks, mesh_crop
        long[:] converged, block
    root_mesh = Mesh1DUniform(boundary_1, boundary_2,
                              boundary_condition_1=bc1,
                              boundary_condition_2=bc2,
                              physical_step=round(step, 9))
    meshes_tree = TreeMesh1DUniform(root_mesh, refinement_coefficient=2, aligned=True)
    while True:
        level = max(meshes_tree.levels)
        converged = np.zeros(len(meshes_tree.__tree[level]), dtype=np.int)
        refinements = []
        for mesh_id, mesh in enumerate(meshes_tree.__tree[level]):
            mesh = dirichlet_poisson_solver_mesh(mesh, f)
            mesh.trim()
            refinement_points_chunks = points_for_refinement(mesh, threshold)
            converged[mesh_id] = np.all(np.array([block.size == 0 for block in refinement_points_chunks]))
            if converged[mesh_id]:
                continue
            elif level < max_level:
                for block in refinement_points_chunks:
                    idx1, idx2, mesh_crop = adjust_range(block, mesh.__num - 1, crop=[10, 10],
                                                         step_scale=meshes_tree.refinement_coefficient)
                    refinements.append(Mesh1DUniform(
                        mesh.to_physical(np.array([mesh.__local_nodes[idx1]]))[0],
                        mesh.to_physical(np.array([mesh.__local_nodes[idx2]]))[0],
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
