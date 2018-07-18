from __future__ import division, print_function
import numpy as np
from scipy.sparse import dia_matrix
from scipy.interpolate import interp1d

from BDMesh.Mesh1DUniform cimport Mesh1DUniform

from libc.math cimport floor, ceil
from cython cimport boundscheck, wraparound


cdef double trapz_1d(double[:] y, double[:] x):
    cdef:
        int nx = x.shape[0], ny = y.shape[0], i
        double res = 0.0
    if nx == ny:
        with boundscheck(False), wraparound(False):
            for i in range(nx - 1):
                res += (x[i + 1] - x[i]) * (y[i + 1] + y[i]) / 2
        return res
    else:
        raise ValueError('x and y must be the same size')


cpdef fd_d2_matrix(int size):
    """
    d2 finite difference matrix generator
    :param size: size of matrix to generate (int)
    :return: d2 finite difference sparse matrix of three diagonals of O(h2) precision.
    """
    cdef:
        double[:] a, b
    a = -2 * np.ones(size)
    b = np.ones(size)
    return dia_matrix(([b, a, b], [-1, 0, 1]), (size, size), dtype=np.double).tocsr()


cpdef interp_fn(double[:] x, double[:] y, str extrapolation='linear'):
    """
    scipy interp1d wrapper to simplify usage
    :param x: 1D array of x nodes values
    :param y: 1D array of the same size as x of interpolated function values at x nodes
    :param extrapolation: extrapolation style could be one of 'linear', 'last', and 'zero'
    :return: function of single argument x which interpolates given input data [x, y]
    """
    if extrapolation == 'linear':
        return interp1d(x, y, bounds_error=False, fill_value='extrapolate')
    elif extrapolation == 'last':
        return interp1d(x, y, bounds_error=False, fill_value=(y[0], y[-1]))
    elif extrapolation == 'zero':
        return interp1d(x, y, bounds_error=False, fill_value=0.0)
    else:
        return interp1d(x, y, bounds_error=False, fill_value=np.nan)


cpdef list points_for_refinement(Mesh1DUniform mesh, double threshold):
    """
    returns sorted arrays of mesh nodes indices, which require refinement
    :param mesh: mesh of type BDMesh.Mesh1DUniform
    :param threshold: threshold value for mesh.residual
    :return: arrays of bad nodes indices for refinement
    """
    cdef:
        long[:] bad_nodes = np.sort(np.where(abs(mesh.residual) > threshold)[0])
        long[:] split_idx = np.where(np.asarray(bad_nodes[1:]) - np.asarray(bad_nodes[:-1]) > 1)[0] + 1
    return list(np.split(bad_nodes, split_idx))


cpdef adjust_range(long[:] idx_range, int max_index, crop=(0, 0), int step_scale=1):
    """
    Calculates start and stop indices for refinement mesh generation given a range of indices
    :param idx_range: 1D array of sorted indices of nodes to build refinement mesh on
    :param max_index: max allowed index
    :param crop: a tuple of two integers (number of nodes of final mesh to be appended for later crop
    :param step_scale: step scale coefficient between original and generated meshes
    :return: a pair of indices and a crop tuple for refinement mesh constructor
    """
    cdef:
        int idx1, idx2
        int[2] mesh_crop = [0, 0]
    idx1 = max(idx_range[0], 0)
    idx2 = min(idx_range[-1], max_index)
    if idx2 - idx1 < 2:
        if idx2 == max_index and idx1 == 0:
            return idx1, idx2, mesh_crop
        if idx1 == 0 and idx2 < max_index:
            idx2 += 1
        elif idx2 == max_index and idx1 > 0:
            idx1 -= 1
        else:
            idx1 -= 1
            idx2 += 1
    if (idx1 - ceil(crop[0] / step_scale)) >= 0:
        mesh_crop[0] = int(ceil(crop[0] / step_scale) * step_scale)
    else:
        mesh_crop[0] = int(floor(idx1 * step_scale))
    idx1 -= int(ceil(mesh_crop[0] / step_scale))
    if (idx2 + ceil(crop[1] / step_scale)) <= max_index:
        mesh_crop[1] = int(ceil(crop[1] / step_scale) * step_scale)
    else:
        mesh_crop[1] = int(floor((max_index - idx2) * step_scale))
    idx2 += int(ceil(mesh_crop[1] / step_scale))
    return idx1, idx2, mesh_crop
