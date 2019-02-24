import numpy as np

from libc.math cimport fabs, ceil, round
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
        double abs_threshold = fabs(threshold)
    for i in range(n):
        if fabs(mesh.__residual[i]) > abs_threshold:
            if i - last > 1:
                result += 1
            last = i
    return result


@boundscheck(False)
@wraparound(False)
cdef int[:, :] refinement_points(Mesh1DUniform mesh, double threshold,
                                 int crop_l=0, int crop_r=0, double step_scale=1.0):
    cdef:
        double abs_threshold = fabs(threshold)
        int i, j = 0, last = -2, n = mesh.num, chunks = refinement_chunks(mesh, abs_threshold)
        int idx_tmp, crop_tmp
        int[2] crop = [<int> ceil(crop_l / step_scale), <int> ceil(crop_r / step_scale)]
        int[:, :] result = np.empty((chunks, 4), dtype=np.int32)
        bint closed = True
    for i in range(n):
        if fabs(mesh.__residual[i]) > abs_threshold:
            if i - last > 1:
                closed = False
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
            closed = True
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
    if not closed:
        closed = True
        result[j, 1] = n - 1
        result[j, 3] = 0
        j += 1
    print('CHUNKS:', chunks, j)
    if result[0][0] == 1:
            result[0][0] = 0
    for i in range(j):
        if result[i][1] - result[i][0] == 0:
            if result[i][0] > 0:
                result[i][0] -= 1
            else:
                result[i][1] += 1
        if result[i][1] - result[i][0] == (result[i][2] + result[i][3]) / step_scale:
            if result[i][2] > 0:
                result[i][2] -= 1
            if result[i][3] > 0:
                result[i][3] -= 1
    return result[:j]
