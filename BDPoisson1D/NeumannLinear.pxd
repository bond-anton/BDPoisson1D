from .Function cimport Function

cpdef neumann_poisson_solver_arrays(double[:] nodes, double[:] f_nodes,
                                    double bc1, double bc2, double j=*, double y0=*)
cpdef neumann_poisson_solver(double[:] nodes, Function f, double bc1, double bc2, double j=*, double y0=*)
