from .Function cimport Function


cpdef dirichlet_poisson_solver_arrays(double[:] nodes, double[:] f_nodes, double bc1, double bc2, double j=*)
cpdef dirichlet_poisson_solver(double[:] nodes, Function f, double bc1, double bc2, double j=*)
cpdef dirichlet_poisson_solver_mesh_arrays(mesh, double[:] f_nodes)
cpdef dirichlet_poisson_solver_mesh(mesh, Function f)
cpdef dirichlet_poisson_solver_amr(double boundary_1, double boundary_2, double step, Function f,
                                   double bc1, double bc2,
                                   double threshold=*, int max_level=*)