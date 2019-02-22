import numpy as np
from scipy.sparse import dia_matrix

from libc.math cimport floor, ceil, round
from cython cimport boundscheck, wraparound
from cpython.array cimport array, clone

from BDMesh.Mesh1DUniform cimport Mesh1DUniform


@boundscheck(False)
@wraparound(False)
cdef double trapz_1d(double[:] y, double[:] x):
    cdef:
        int nx = x.shape[0], ny = y.shape[0], i
        double res = 0.0
    for i in range(nx - 1):
        res += (x[i + 1] - x[i]) * (y[i + 1] + y[i]) / 2
    return res


@boundscheck(False)
@wraparound(False)
cdef double[:] gradient1d(double[:] y, double[:] x, int n):
    cdef:
        int i
        double a, b, c, dx1, dx2
        array[double] result, template = array('d')
    result = clone(template, n, zero=False)
    dx1 = x[1] - x[0]
    dx2 = x[2] - x[1]
    a = -(2. * dx1 + dx2)/(dx1 * (dx1 + dx2))
    b = (dx1 + dx2) / (dx1 * dx2)
    c = - dx1 / (dx2 * (dx1 + dx2))
    result[0] = a * y[0] + b * y[1] + c * y[2]
    dx1 = x[n - 2] - x[n - 3]
    dx2 = x[n - 1] - x[n - 2]
    a = dx2 / (dx1 * (dx1 + dx2))
    b = - (dx2 + dx1) / (dx1 * dx2)
    c = (2.0 * dx2 + dx1) / (dx2 * (dx1 + dx2))
    result[n - 1] = a * y[n - 3] + b * y[n - 2] + c * y[n - 1]
    for i in range(1, n - 1):
        result[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
    return result


@boundscheck(False)
@wraparound(False)
cdef int refinement_chunks(Mesh1DUniform mesh, double threshold):
    cdef:
        int i, last = -2, n = mesh.num, result = 0
    for i in range(n):
        if mesh.__residual[i] > threshold:
            if i - last > 1:
                result += 1
            last = i
    return result


@boundscheck(False)
@wraparound(False)
cdef int[:, :] refinement_points(Mesh1DUniform mesh, double threshold,
                                 int crop_l=0, int crop_r=0, double step_scale=1.0):
    cdef:
        int i, j = 0, last = -2, n = mesh.num, chunks = refinement_chunks(mesh, threshold)
        int idx_tmp, crop_tmp
        int[2] crop = [<int> ceil(crop_l / step_scale), <int> ceil(crop_r / step_scale)]
        int[:, :] result = np.empty((chunks, 4), dtype=np.int32)
    for i in range(n):
        if mesh.__residual[i] > threshold:
            if i - last > 1:
                idx_tmp = i - crop[0]
                crop_tmp = crop[0]
                if idx_tmp < 0:
                    idx_tmp = 0
                    crop_tmp = i
                if j > 0 and idx_tmp <= result[j - 1, 1]:
                    j -= 1
                else:
                    result[j, 0] = idx_tmp
                    result[j, 2] = <int> round(crop_tmp * step_scale)
            last = i

        elif i - last == 1:
            idx_tmp = last + crop[1]
            crop_tmp = crop[1]
            if idx_tmp > n-1:
                crop_tmp = n - last - 1
                idx_tmp = n - 1
            result[j, 1] = idx_tmp
            result[j, 3] = <int> round(crop_tmp * step_scale)
            j += 1
            if idx_tmp == n - 1:
                break
    if j < chunks:
        result[j, 1] = n - 1
        result[j, 3] = 0
    if result[0][0] == 1:
            result[0][0] = 0
    for j in range(chunks):
        if result[j][1] - result[j][0] == 0:
            if result[j][0] > 0:
                result[j][0] -= 1
            else:
                result[j][1] += 1
        if result[j][1] - result[j][0] == (result[j][2] + result[j][3]) / step_scale:
            if result[j][2] > 0:
                result[j][2] -= 1
            if result[j][3] > 0:
                result[j][3] -= 1
    return result
