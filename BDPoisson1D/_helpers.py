from __future__ import division, print_function
import math as m
import numpy as np
from scipy import sparse
from scipy.interpolate import interp1d

from BDMesh import Mesh1DUniform


def fd_d2_matrix(size):
    """
    d2 finite difference matrix generator
    :param size: size of matrix to generate (int)
    :return: d2 finite difference sparse matrix of three diagonals of O(h2) precision.
    """
    a = -2 * np.ones(size)
    b = np.ones(size - 1)
    c = np.ones(size - 1)
    return sparse.diags([c, a, b], offsets=[-1, 0, 1], shape=(size, size), format='csc')


def interp_fn(x, f, extrapolation='linear'):
    """
    scipy interp1d wrapper to simplify usage
    :param x: 1D array of x nodes values
    :param f: 1D array of the same size as x of interpolated function values at x nodes
    :param extrapolation: extrapolation style could be one of 'linear', 'last', and 'zero'
    :return: function of single argument x which interpolates given input data [x, f]
    """
    if extrapolation == 'linear':
        fill_value = 'extrapolate'
    elif extrapolation == 'last':
        fill_value = (f[0], f[-1])
    elif extrapolation == 'zero':
        fill_value = 0.0
    else:
        fill_value = np.nan
    interpolator = interp1d(x, f, bounds_error=False, fill_value=fill_value)
    return interpolator


def points_for_refinement(mesh, threshold):
    assert isinstance(mesh, Mesh1DUniform)
    assert isinstance(threshold, (float, int))
    bad_nodes = np.sort(np.where(abs(mesh.residual) > threshold)[0])
    split_idx = np.where(bad_nodes[1:] - bad_nodes[:-1] > 1)[0] + 1
    bad_nodes = np.split(bad_nodes, split_idx)
    return bad_nodes


def adjust_range(idx_range, max_index, crop=None, step_scale=1):
    idx1 = idx_range[0] if idx_range[0] >= 0 else 0
    idx2 = idx_range[-1] if idx_range[-1] <= max_index else max_index
    mesh_crop = [0, 0]
    if idx2 - idx1 < 2:
        if idx1 == 0 and idx2 != max_index:
            idx2 += 1
        elif idx2 == max_index and idx1 != 0:
            idx1 -= 1
        elif idx2 == max_index and idx1 == 0:
            raise ValueError('the range is too short!')
        else:
            idx1 -= 1
            idx2 += 1
    if (idx1 - m.floor(step_scale * crop[0])) >= 0:
        mesh_crop[0] = int(crop[0])
    else:
        mesh_crop[0] = int(m.floor(idx1 / step_scale))
    idx1 -= int(m.floor(step_scale * mesh_crop[0]))
    if (idx2 + m.ceil(step_scale * crop[1])) <= max_index:
        mesh_crop[1] = int(crop[1])
    else:
        mesh_crop[1] = int(m.floor((max_index - idx2) / step_scale))
    idx2 += int(m.floor(step_scale * mesh_crop[1]))
    return idx1, idx2, mesh_crop
