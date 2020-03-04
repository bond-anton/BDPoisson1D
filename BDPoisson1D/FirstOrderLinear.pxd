from .Function cimport Function
from BDMesh.Mesh1DUniform cimport Mesh1DUniform


cpdef double[:] dirichlet_first_order_solver_arrays(double[:] nodes, double[:] p_nodes, double[:] f_nodes,
                                                       double bc1, double bc2, double j=*)
cpdef double[:] dirichlet_first_order_solver(double[:] nodes, Function p, Function f,
                                                double bc1, double bc2, double j=*)
cpdef void dirichlet_first_order_solver_mesh_arrays(Mesh1DUniform mesh, double[:] p_nodes, double[:] f_nodes)
cpdef void dirichlet_first_order_solver_mesh(Mesh1DUniform mesh, Function p, Function f)
