from BDMesh.Mesh1DUniform cimport Mesh1DUniform
from .Function cimport Function, Functional

cpdef dirichlet_non_linear_poisson_solver_arrays(double[:] nodes, double[:] y0_nodes,
                                                 double[:] f_nodes, double[:] df_ddy_nodes,
                                                 double bc1, double bc2, double j=*, double w=*)

cpdef dirichlet_non_linear_poisson_solver(double[:] nodes, Function y0, Functional f, Functional df_ddy,
                                          double bc1, double bc2, double j=*, double w=*)

cpdef dirichlet_non_linear_poisson_solver_mesh_arrays(Mesh1DUniform mesh,
                                                      double[:] y0_nodes, double[:] f_nodes,
                                                      double[:] df_ddy_nodes, double w=*)

cpdef dirichlet_non_linear_poisson_solver_mesh(Mesh1DUniform mesh, Function y0, Functional f, Functional df_ddy,
                                               double w=*)

cpdef dirichlet_non_linear_poisson_solver_recurrent_mesh(Mesh1DUniform mesh,
                                                         Function y0, Functional f, Functional df_ddy,
                                                         int max_iter=*, double threshold=*)

cpdef dirichlet_non_linear_poisson_solver_amr(double boundary_1, double boundary_2, double step, Function y0,
                                              Functional f, Functional df_ddy, double bc1, double bc2,
                                              int max_iter=*,
                                              double residual_threshold=*, double int_residual_threshold=*,
                                              int max_level=*, double mesh_refinement_threshold=*)
