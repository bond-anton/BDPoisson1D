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


def interp_Fn(Z, F, interp_type='linear'):
    """
    Z and F must be 1D arrays of equal size
    interp_type could be one of
    'linear'
    'last'
    'zero'
    """
    # print 'type:', interp_type
    def interp(z):
        interpolator = interp1d(Z, F, bounds_error=True)
        xs = interpolator.x
        ys = interpolator.y

        def pointwise(x):
            if x < xs[0]:
                if interp_type == 'linear':
                    return ys[0] + (x - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
                elif interp_type == 'last':
                    return ys[0]
                elif interp_type == 'zero':
                    return 0.0
            elif x > xs[-1]:
                if interp_type == 'linear':
                    return ys[-1] + (x - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
                elif interp_type == 'last':
                    return ys[-1]
                elif interp_type == 'zero':
                    return 0.0
            else:
                return np.float(interpolator(x))

        if isinstance(z, (np.ndarray, list, tuple)):
            return np.array([pointwise(z_i) for z_i in z], dtype=np.float)
        else:
            return pointwise(z)

    return interp


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
