from BDMesh.Mesh1DUniform cimport Mesh1DUniform
from BDFunction1D cimport Function


cpdef double[:] cauchy_first_order_solver_arrays(double[:] nodes, double[:] f_nodes,
                                                 double ic=*, int idx=*, double j=*)
cpdef Function cauchy_first_order_solver(double[:] nodes, Function f,
                                         double ic=*, int idx=*, double j=*)
cpdef void cauchy_first_order_solver_mesh_arrays(Mesh1DUniform mesh, double[:] f_nodes,
                                                 double ic=*, int idx=*)
cpdef Function cauchy_first_order_solver_mesh(Mesh1DUniform mesh, Function f, double ic=*, int idx=*)

cpdef double[:] dirichlet_first_order_solver_arrays(double[:] nodes, double[:] p_nodes, double[:] f_nodes,
                                                    double bc1, double bc2, double j=*)
cpdef Function dirichlet_first_order_solver(double[:] nodes, Function p, Function f,
                                            double bc1, double bc2, double j=*)
cpdef void dirichlet_first_order_solver_mesh_arrays(Mesh1DUniform mesh, double[:] p_nodes, double[:] f_nodes)
cpdef Function dirichlet_first_order_solver_mesh(Mesh1DUniform mesh, Function p, Function f)
