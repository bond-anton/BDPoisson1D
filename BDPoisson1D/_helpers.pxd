from BDMesh.Mesh1DUniform cimport Mesh1DUniform

cdef double trapz_1d(double[:] y, double[:] x)
cdef double[:] gradient1d(double[:] y, double[:] x, int n)
cdef int refinement_chunks(Mesh1DUniform mesh, double threshold)
cdef int[:, :] refinement_points(Mesh1DUniform mesh, double threshold,
                                 int crop_l=*, int crop_r=*, double step_scale=*)
