import numpy as np

from cython cimport boundscheck, wraparound
from cpython.array cimport array, clone

from scipy.linalg.cython_lapack cimport dgtsv

from BDMesh.TreeMesh1DUniform cimport TreeMesh1DUniform
from BDMesh.Mesh1DUniform cimport Mesh1DUniform
from ._helpers cimport gradient1d, points_for_refinement, adjust_range
from .Function cimport Function


@boundscheck(False)
@wraparound(False)
cpdef double[:, :] dirichlet_poisson_solver_arrays(double[:] nodes, double[:] f_nodes, double bc1, double bc2, double j=1.0):
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
        result: 2D array of solution function y(x) values on nodes array and the error of the solution.
    """
    cdef:
        int i, n = nodes.shape[0] - 2, nrhs = 1, info
        double[:] d2y
        array[double] y, residual, f, d, dl, du, template = array('d')
        double[:, :] result = np.empty((n + 2, 2), dtype=np.double)
    d = clone(template, n, zero=False)
    dl = clone(template, n - 1, zero=False)
    du = clone(template, n - 1, zero=False)
    f = clone(template, n, zero=False)
    result[0, 0] = bc1
    result[n + 1, 0] = bc2
    for i in range(n):
        if i < n - 1:
            dl[i] = 1.0
            du[i] = 1.0
        d[i] = -2.0
        f[i] = (j * (nodes[i + 1] - nodes[i])) ** 2 * f_nodes[i + 1]
    f[0] -= bc1
    f[n - 1] -= bc2
    dgtsv(&n, &nrhs, &dl[0], &d[0], &du[0], &f[0], &n, &info)
    for i in range(n):
        result[i + 1, 0] = f[i]
    d2y = gradient1d(gradient1d(result[:, 0], nodes, n + 2), nodes, n + 2)
    for i in range(n + 2):
        result[i, 1] = f_nodes[i] - d2y[i] / (j * j)
    return result


cpdef double[:, :] dirichlet_poisson_solver(double[:] nodes, Function f, double bc1, double bc2, double j=1.0):
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


cpdef void dirichlet_poisson_solver_mesh_arrays(Mesh1DUniform mesh, double[:] f_nodes):
    """
    Solves 1D differential equation of the form
        d2y/dx2 = f(x)
        y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
    using FDE algorithm of O(h2) precision.

    :param mesh: BDMesh to solve on.
    :param f_nodes: 1D array of values of f(x) on nodes array. Must be same shape as nodes.
    """
    result = dirichlet_poisson_solver_arrays(mesh.__local_nodes, f_nodes,
                                             mesh.__boundary_condition_1, mesh.__boundary_condition_2,
                                             mesh.j())
    mesh.solution = result[:, 0]
    mesh.residual = result[:, 1]
    # return mesh


cpdef void dirichlet_poisson_solver_mesh(Mesh1DUniform mesh, Function f):
    """
    Solves 1D differential equation of the form
        d2y/dx2 = f(x)
        y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
    using FDE algorithm of O(h2) precision.

    :param mesh: BDMesh to solve on.
    :param f: function f(x) callable on nodes array.
    """
    dirichlet_poisson_solver_mesh_arrays(mesh, f.evaluate(mesh.to_physical_coordinate(mesh.__local_nodes)))


@boundscheck(False)
@wraparound(True)
cpdef dirichlet_poisson_solver_amr(double boundary_1, double boundary_2, double step, Function f,
                                   double bc1, double bc2,
                                   int max_iter=1000,
                                   double threshold=1e-2, int max_level=10):
    """
    Linear Poisson equation solver with Adaptive Mesh Refinement algorithm.
    :param boundary_1: physical nodes left boundary.
    :param boundary_2: physical nodes right boundary.
    :param step: physical nodes step.
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
    while i < max_iter:
        i += 1
        level = max(meshes_tree.levels)
        converged = np.zeros(len(meshes_tree.__tree[level]), dtype=np.int)
        refinements = []
        for mesh_id, mesh in enumerate(meshes_tree.__tree[level]):
            dirichlet_poisson_solver_mesh(mesh, f)
            mesh.trim()
            refinement_points_chunks = points_for_refinement(mesh, threshold)
            converged[mesh_id] = not len(refinement_points_chunks) > 0
            if converged[mesh_id]:
                continue
            elif level < max_level:
                for block in refinement_points_chunks:
                    idx1, idx2, mesh_crop = adjust_range(block, mesh.__num - 1, crop=[20, 20],
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
