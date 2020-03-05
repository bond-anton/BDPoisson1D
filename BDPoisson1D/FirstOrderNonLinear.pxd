from .Function cimport Function, Functional
from BDMesh.Mesh1DUniform cimport Mesh1DUniform


cpdef double[:, :] dirichlet_non_linear_first_order_solver_arrays(double[:] nodes, double[:] y0_nodes,
                                                                  double[:] p_nodes,
                                                                  double[:] f_nodes, double[:] df_dy_nodes,
                                                                  double bc1, double bc2, double j=*, double w=*)
cpdef double[:, :] dirichlet_non_linear_first_order_solver(double[:] nodes, Function y0, Function p,
                                                           Functional f, Functional df_dy,
                                                           double bc1, double bc2, double j=*, double w=*)
cpdef void dirichlet_non_linear_first_order_solver_mesh_arrays(Mesh1DUniform mesh, double[:] y0_nodes,
                                                               double[:] p_nodes,
                                                               double[:] f_nodes, double[:] df_dy_nodes, double w=*)
cpdef void dirichlet_non_linear_first_order_solver_mesh(Mesh1DUniform mesh, Function y0, Function p,
                                                        Functional f, Functional df_dy,
                                                        double w=*)
cpdef void dirichlet_non_linear_first_order_solver_recurrent_mesh(Mesh1DUniform mesh, Function y0, Function p,
                                                                  Functional f, Functional df_dy, double w=*,
                                                                  int max_iter=*, double threshold=*)
