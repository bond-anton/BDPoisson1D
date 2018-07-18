from BDMesh.Mesh1DUniform cimport Mesh1DUniform

cdef double trapz_1d(double[:] y, double[:] x)
cpdef fd_d2_matrix(int size)
cpdef interp_fn(double[:] x, double[:] y, str extrapolation=*)
cpdef list points_for_refinement(Mesh1DUniform mesh, double threshold)
cpdef adjust_range(long[:] idx_range, int max_index, crop=*, int step_scale=*)
