import warnings
import numpy as np

from cython cimport boundscheck, wraparound
from cpython.array cimport array, clone

from scipy.linalg.cython_lapack cimport dgtsv
from libc.math cimport fabs

from BDMesh.Mesh1DUniform cimport Mesh1DUniform
from BDMesh.TreeMesh1DUniform cimport TreeMesh1DUniform
from BDFunction1D cimport Function

from._helpers cimport trapz_1d, gradient1d, refinement_points


@boundscheck(False)
@wraparound(False)
cpdef double[:, :] neumann_poisson_solver_arrays(double[:] nodes, double[:] f_nodes,
                                                 double bc1, double bc2, double j=1.0, double y0=0.0):
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
    cdef:
        int i, n = nodes.size, ns = n - 1, nrhs = 1, info
        double integral = trapz_1d(f_nodes, nodes)
        array[double] f, d, dl, du, template = array('d')
        double[:] d2y
        double[:, :] result = np.empty((n, 2), dtype=np.double)
    if fabs(integral - bc2 + bc1) > 1e-4:
        warnings.warn('Not well-posed! Redefine f function and boundary conditions or refine the mesh!')
    d = clone(template, n, zero=False)
    dl = clone(template, n - 2, zero=False)
    du = clone(template, n - 2, zero=False)
    f = clone(template, n, zero=False)
    result[0, 0] = bc1
    result[n - 1, 0] = bc2
    for i in range(n-1):
        if i < n - 2:
            dl[i] = 1.0
            du[i] = 1.0
        d[i] = -2.0
        f[i] = (j * (nodes[i + 1] - nodes[i])) ** 2 * f_nodes[i + 1]
    f[0] += (nodes[1] - nodes[0]) ** 2 * f_nodes[0] + 2 * (nodes[1] - nodes[0]) * bc1 + y0
    f[n - 1] -= 2 * (nodes[n - 1] - nodes[n - 2]) * bc2
    d[0] = 0
    dl[n - 3] = 2
    dgtsv(&ns, &nrhs, &dl[0], &d[0], &du[0], &f[0], &ns, &info)
    for i in range(n):
        result[i + 1, 0] = f[i]
    result[0, 0] = y0
    d2y = gradient1d(gradient1d(result[:, 0], nodes), nodes)
    for i in range(n):
        result[i, 1] = f_nodes[i] - d2y[i] / (j * j)
    return result


@boundscheck(False)
@wraparound(True)
cpdef double[:, :] neumann_poisson_solver(double[:] nodes, Function f,
                                          double bc1, double bc2, double j=1.0, double y0=0.0):
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
    return neumann_poisson_solver_arrays(nodes, f.evaluate(nodes), bc1, bc2, j, y0)


@boundscheck(False)
@wraparound(True)
cpdef void neumann_poisson_solver_mesh_arrays(Mesh1DUniform mesh, double[:] f_nodes):
    cdef:
        double[:, :] result
    result = neumann_poisson_solver_arrays(mesh.__local_nodes, f_nodes,
                                           mesh.__boundary_condition_1, mesh.__boundary_condition_2,
                                           mesh.j(), mesh.__solution[0])
    mesh.__solution = result[:, 0]
    mesh.__residual = result[:, 1]


@boundscheck(False)
@wraparound(True)
cpdef void neumann_poisson_solver_mesh(Mesh1DUniform mesh, Function f):
    neumann_poisson_solver_mesh_arrays(mesh, f.evaluate(mesh.physical_nodes))


@boundscheck(False)
@wraparound(True)
cpdef void neumann_poisson_solver_mesh_amr(TreeMesh1DUniform meshes_tree, Function f,
                                           int max_iter=1000, double threshold=1.0e-2, int max_level=10):
    cdef:
        int level, i = 0, j, converged, n
        Mesh1DUniform mesh
        int[:, :] refinements
        Mesh1DUniform refinement_mesh
        double[:] dy
    while i < max_iter:
        i += 1
        level = max(meshes_tree.levels)
        converged = 0
        n = 0
        for mesh in meshes_tree.__tree[level]:
            n += 1
            neumann_poisson_solver_mesh(mesh, f)
            mesh.trim()
            dy = gradient1d(mesh.__solution, mesh.__local_nodes)
            refinements = refinement_points(mesh, threshold, crop_l=100, crop_r=100,
                                            step_scale=meshes_tree.refinement_coefficient)
            if refinements.shape[0] == 0:
                converged += 1
                continue
            if level < max_level and i < max_iter:
                for j in range(refinements.shape[0]):
                    refinement_mesh = Mesh1DUniform(
                        mesh.__physical_boundary_1 + mesh.j() * mesh.__local_nodes[refinements[j][0]],
                        mesh.__physical_boundary_1 + mesh.j() * mesh.__local_nodes[refinements[j][1]],
                        boundary_condition_1=dy[refinements[j][0]] / mesh.j(),
                        boundary_condition_2=dy[refinements[j][1]] / mesh.j(),
                        physical_step=mesh.physical_step/meshes_tree.refinement_coefficient,
                        crop=[refinements[j][2], refinements[j][3]])
                    refinement_mesh.__solution[0] = mesh.__solution[refinements[j][0]]
                    meshes_tree.add_mesh(refinement_mesh)
        meshes_tree.remove_coarse_duplicates()
        if converged == n or level == max_level:
            break


@boundscheck(False)
@wraparound(True)
cpdef TreeMesh1DUniform neumann_poisson_solver_amr(double boundary_1, double boundary_2, double step, Function f,
                                                   double bc1, double bc2, double y0=0.0,
                                                   int max_iter=1000, double threshold=1.0e-2, int max_level=10):
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
    root_mesh.__solution = np.ones(root_mesh.num, dtype=np.double) * y0
    meshes_tree = TreeMesh1DUniform(root_mesh, refinement_coefficient=2, aligned=True)
    neumann_poisson_solver_mesh_amr(meshes_tree, f, max_iter, threshold, max_level)
    return meshes_tree
