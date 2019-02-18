from .Function cimport Function
from BDMesh.Mesh1DUniform cimport Mesh1DUniform

cpdef double[:, :] dirichlet_poisson_solver_arrays(double[:] nodes, double[:] f_nodes,
                                                   double bc1, double bc2, double j=*)
cpdef double[:, :] dirichlet_poisson_solver(double[:] nodes, Function f, double bc1, double bc2, double j=*)
cpdef void dirichlet_poisson_solver_mesh_arrays(Mesh1DUniform mesh, double[:] f_nodes)
cpdef void dirichlet_poisson_solver_mesh(Mesh1DUniform mesh, Function f)
cpdef dirichlet_poisson_solver_amr(double boundary_1, double boundary_2, double step, Function f,
                                   double bc1, double bc2,
                                   int max_iter=*,
                                   double threshold=*, int max_level=*)