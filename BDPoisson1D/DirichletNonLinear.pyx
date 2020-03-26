import numpy as np

from cython cimport boundscheck, wraparound
from cpython.array cimport array, clone

from scipy.linalg.cython_lapack cimport dgtsv

from BDMesh.Mesh1DUniform cimport Mesh1DUniform
from BDMesh.TreeMesh1DUniform cimport  TreeMesh1DUniform
from BDFunction1D cimport Function
from BDFunction1D.Functional cimport Functional
from BDFunction1D.Interpolation cimport InterpolateFunction, InterpolateFunctionMesh
from ._helpers cimport gradient1d, refinement_points


@boundscheck(False)
@wraparound(False)
cpdef double[:, :] dirichlet_non_linear_poisson_solver_arrays(double[:] nodes, double[:] y0_nodes,
                                                              double[:] f_nodes, double[:] df_dy_nodes,
                                                              double bc1, double bc2, double j=1.0, double w=1.0):
    """
    Solves nonlinear 1D differential equation of the form
        d2y/dx2 = f(x, y(x))
        y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
    using FDE algorithm of O(h2) precision and Tailor series for linearization.
        y = y0 + Dy
        f(x, y(x)) ~= f(x, y0) + df/dy(x, y=y0)*Dy
    :param nodes: 1D array of x nodes. Must include boundary points.
    :param y0_nodes: 1D array of y(x) initial approximation at nodes. Must be same shape as nodes.
    :param f_nodes: 1D array of values of f(x) on nodes array. Must be same shape as nodes.
    :param df_dy_nodes: df/dy(x, y=y0).
    :param bc1: boundary condition at nodes[0] point (a number).
    :param bc2: boundary condition at nodes[n] point (a number).
    :param j: Jacobian.
    :param w: the weight for Dy (default w=1.0).
    :return: solution y = y0 + w * Dy; Dy; residual.
    """
    cdef:
        int i, n = nodes.shape[0] - 2, nrhs = 1, info
        array[double] b, f, d, dl, du, template = array('d')
        double[:, :] result = np.empty((n + 2, 3), dtype=np.double)
    d = clone(template, n, zero=False)
    dl = clone(template, n - 1, zero=False)
    du = clone(template, n - 1, zero=False)
    b = clone(template, n, zero=False)
    f = clone(template, n, zero=False)
    for i in range(1, n - 1):
        b[i] = y0_nodes[i] - 2.0 * y0_nodes[i + 1] + y0_nodes[i + 2]
    b[0] = - 2.0 * y0_nodes[1] + y0_nodes[2]
    b[n - 1] = y0_nodes[n - 1] -2.0 * y0_nodes[n]
    for i in range(n):
        if i < n - 1:
            dl[i] = 1.0
            du[i] = 1.0
        d[i] = -2.0 - (j * (nodes[i + 1] - nodes[i])) ** 2 * df_dy_nodes[i + 1]
        f[i] = (j * (nodes[i + 1] - nodes[i])) ** 2 * f_nodes[i + 1] - b[i]
    f[0] -= bc1
    f[n - 1] -= bc2
    result[0, 1] = bc1 - y0_nodes[0]
    result[0, 0] = y0_nodes[0] + w * result[0, 1]
    result[n + 1, 1] = bc2 - y0_nodes[n + 1]
    result[n + 1, 0] = y0_nodes[n + 1] + w * result[n + 1, 1]
    dgtsv(&n, &nrhs, &dl[0], &d[0], &du[0], &f[0], &n, &info)
    for i in range(1, n + 1):
        result[i, 1] = f[i - 1]
        result[i, 0] = y0_nodes[i] + w * result[i, 1]
    d2_y0 = gradient1d(gradient1d(y0_nodes, nodes), nodes)
    d2_dy = gradient1d(gradient1d(result[:, 1], nodes), nodes)
    for i in range(n + 2):
        result[i, 2] = f_nodes[i] - (d2_dy[i] + d2_y0[i]) / (j * j)
    return result


@boundscheck(False)
@wraparound(False)
cpdef Function dirichlet_non_linear_poisson_solver(double[:] nodes, Function y0, Functional f, Functional df_dy,
                                                   double bc1, double bc2, double j=1.0, double w=1.0):
    """
    Solves nonlinear 1D differential equation of the form
        d2y/dx2 = f(x, y(x))
        y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
    using FDE algorithm of O(h2) precision and Tailor series for linearization.
        y = y0 + Dy
        f(x, y(x)) ~= f(x, y0) + df/dy(x, y=y0)*Dy
    :param nodes: 1D array of x nodes. Must include boundary points.
    :param y0: callable of y(x) initial approximation.
    :param f: callable of f(x) to be evaluated on nodes array.
    :param df_dy: callable for evaluation of df/dy(x, y=y0).
    :param bc1: boundary condition at nodes[0] point (a number).
    :param bc2: boundary condition at nodes[n] point (a number).
    :param j: Jacobian.
    :param w: the weight for Dy (default w=1.0).
    :return: solution as callable function y = y0 + w * Dy; Dy; residual.
    """
    cdef:
        int i, n = nodes.shape[0]
        double[:] phys_nodes = clone(array('d'), n, zero=False)
        double[:, :] y
    y = dirichlet_non_linear_poisson_solver_arrays(nodes, y0.evaluate(nodes),
                                                   f.evaluate(nodes), df_dy.evaluate(nodes),
                                                   bc1, bc2, j, w)
    for i in range(n):
        phys_nodes[i] = j * nodes[i]
    return InterpolateFunction(phys_nodes, y[:, 0], y[:, 2])


@boundscheck(False)
@wraparound(False)
cpdef void dirichlet_non_linear_poisson_solver_mesh_arrays(Mesh1DUniform mesh,
                                                           double[:] y0_nodes, double[:] f_nodes,
                                                           double[:] df_dy_nodes, double w=1.0):
    """
    Solves nonlinear 1D differential equation of the form
        d2y/dx2 = f(x, y(x))
        y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
    using FDE algorithm of O(h2) precision and Tailor series for linearization.
        y = y0 + Dy
        f(x, y(x)) ~= f(x, y0) + df/dy(x, y=y0)*Dy
    :param mesh: 1D Uniform mesh with boundary conditions and Jacobian.
    :param y0_nodes: 1D array of y(x) initial approximation at mesh nodes.
    :param f_nodes: 1D array of values of f(x) on mesh nodes.
    :param df_dy_nodes: df/dy(x, y=y0).
    :param w: the weight for Dy (default w=1.0)
    :return: mesh with solution y = y0 + w * Dy, and residual; Dy.
    """
    cdef:
        double[:, :] result
    result = dirichlet_non_linear_poisson_solver_arrays(mesh.__local_nodes, y0_nodes, f_nodes,
                                                        df_dy_nodes,
                                                        mesh.__boundary_condition_1,
                                                        mesh.__boundary_condition_2,
                                                        mesh.j(), w)
    mesh.solution = result[:, 0]
    mesh.residual = result[:, 2]


@boundscheck(False)
@wraparound(False)
cpdef Function dirichlet_non_linear_poisson_solver_mesh(Mesh1DUniform mesh, Function y0,
                                                        Functional f, Functional df_dy, double w=1.0):
    """
    Solves nonlinear 1D differential equation of the form
        d2y/dx2 = f(x, y(x))
        y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
    using FDE algorithm of O(h2) precision and Tailor series for linearization.
        y = y0 + Dy
        f(x, y(x)) ~= f(x, y0) + df/dy(x, y=y0)*Dy
    :param mesh: 1D Uniform mesh with boundary conditions and Jacobian.
    :param y0: callable of y(x) initial approximation.
    :param f: callable of f(x) to be evaluated on nodes array.
    :param df_dy: callable for evaluation of df/dy(x, y=y0).
    :param w: the weight for Dy (default w=1.0)
    :return: mesh with solution y = y0 + w * Dy, and residual; callable solution function; Dy.
    """
    cdef:
        double[:] physical_nodes = mesh.to_physical_coordinate(mesh.__local_nodes)
    dirichlet_non_linear_poisson_solver_mesh_arrays(mesh,
                                                    y0.evaluate(physical_nodes),
                                                    f.evaluate(physical_nodes),
                                                    df_dy.evaluate(physical_nodes), w)
    return InterpolateFunctionMesh(mesh)


@boundscheck(False)
@wraparound(False)
cpdef Function dirichlet_non_linear_poisson_solver_recurrent_mesh(Mesh1DUniform mesh,
                                                                  Function y0, Functional f, Functional df_dy,
                                                                  int max_iter=1000, double threshold=1e-7):
    """
    Solves nonlinear 1D differential equation of the form
        d2y/dx2 = f(x, y(x))
        y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
    using FDE algorithm of O(h2) precision and Tailor series for linearization.
        y = y0 + Dy
        f(x, y(x)) ~= f(x, y0) + df/dy(x, y=y0)*Dy
    Recurrent successive approximation of y0 is used to achieve given residual error threshold value.
    :param mesh: 1D Uniform mesh with boundary conditions and Jacobian.
    :param y0: callable of y(x) initial approximation.
    :param f: callable of f(x) to be evaluated on nodes array.
    :param df_dy: callable for evaluation of df/dy(x, y=y0).
    :param max_iter: maximal number of allowed iterations.
    :param threshold: convergence residual error threshold.
    :return: mesh with solution y = y0 + w * Dy, and residual; callable solution function.
    """
    cdef:
        int i
    for i in range(max_iter):
        y0 = dirichlet_non_linear_poisson_solver_mesh(mesh, y0, f, df_dy)
        if abs(mesh.integrational_residual) <= threshold:  # or max(abs(np.asarray(dy))) <= 2 * np.finfo(np.float).eps:
            break
        f.__f = y0
        df_dy.__f = y0
    return y0


@boundscheck(False)
@wraparound(False)
cpdef Function dirichlet_non_linear_poisson_solver_mesh_amr(TreeMesh1DUniform meshes_tree,
                                                            Function y0, Functional f, Functional df_dy,
                                                            int max_iter=1000,
                                                            double residual_threshold=1e-7,
                                                            double int_residual_threshold=1e-6,
                                                            int max_level=20, double mesh_refinement_threshold=1e-7):
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
            y0 = dirichlet_non_linear_poisson_solver_recurrent_mesh(mesh, y0, f, df_dy,
                                                                    max_iter, int_residual_threshold)
            f.__f = y0
            df_dy.__f = y0
            mesh.trim()
            refinements = refinement_points(mesh, residual_threshold, crop_l=20, crop_r=20,
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
    return InterpolateFunctionMesh(meshes_tree)


@boundscheck(False)
@wraparound(False)
cpdef Function dirichlet_non_linear_poisson_solver_amr(double boundary_1, double boundary_2, double step,
                                                       Function y0, Functional f, Functional df_dy,
                                                       double bc1, double bc2,
                                                       int max_iter=1000,
                                                       double residual_threshold=1e-3,
                                                       double int_residual_threshold=1e-6,
                                                       int max_level=20,
                                                       double mesh_refinement_threshold=1e-7):
    """
        Solves nonlinear 1D differential equation of the form
            d2y/dx2 = f(x, y(x))
            y(x0) = bc1, y(xn) = bc2 (Dirichlet boundary condition)
        using FDE algorithm of O(h2) precision and Tailor series for linearization.
            y = y0 + Dy
            f(x, y(x)) ~= f(x, y0) + df/dy(x, y=y0)*Dy
        Recurrent successive approximation of y0 and adaptive mesh refinement
        algorithms are used to achieve given residual error threshold value.
        :param boundary_1: physical nodes left boundary.
        :param boundary_2: physical nodes right boundary.
        :param step: physical nodes step.
        :param y0: callable of y(x) initial approximation.
        :param f: callable of f(x) to be evaluated on nodes array.
        :param df_dy: callable for evaluation of df/dy(x, y=y0).
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
    return dirichlet_non_linear_poisson_solver_mesh_amr(meshes_tree, y0, f, df_dy,
                                                        max_iter, residual_threshold, int_residual_threshold,
                                                        max_level, mesh_refinement_threshold)
