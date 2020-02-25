from .Function cimport Function
from BDMesh.Mesh1DUniform cimport Mesh1DUniform
from BDMesh.TreeMesh1DUniform cimport TreeMesh1DUniform

cpdef double[:, :] dirichlet_first_order_solver_arrays(double[:] nodes, double[:] p_nodes, double[:] f_nodes,
                                                       double ic, double j=*)
