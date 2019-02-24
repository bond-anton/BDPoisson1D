from .Function cimport Function
from BDMesh.Mesh1DUniform cimport Mesh1DUniform
from BDMesh.TreeMesh1DUniform cimport TreeMesh1DUniform

cpdef double[:, :] neumann_poisson_solver_arrays(double[:] nodes, double[:] f_nodes,
                                    double bc1, double bc2, double j=*, double y0=*)
cpdef double[:, :] neumann_poisson_solver(double[:] nodes, Function f, double bc1, double bc2, double j=*, double y0=*)
cpdef void neumann_poisson_solver_mesh_arrays(Mesh1DUniform mesh, double[:] f_nodes)
cpdef void neumann_poisson_solver_mesh(Mesh1DUniform mesh, Function f)
cpdef void neumann_poisson_solver_mesh_amr(TreeMesh1DUniform meshes_tree, Function f,
                                           int max_iter=*, double threshold=*, int max_level=*)
cpdef TreeMesh1DUniform neumann_poisson_solver_amr(double boundary_1, double boundary_2, double step, Function f,
                                                   double bc1, double bc2, double y0=*,
                                                   int max_iter=*, double threshold=*, int max_level=*)
