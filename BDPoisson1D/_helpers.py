from __future__ import division, print_function
import numpy as np
from scipy.interpolate import interp1d


def interp_Fn(Z, F, interp_type='linear'):
    """
    z and F must be 1D arrays of equal size
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
                    # print 'here'
                    return ys[0]
                elif interp_type == 'zero':
                    return 0.0
            elif x > xs[-1]:
                if interp_type == 'linear':
                    return ys[-1] + (x - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
                elif interp_type == 'last':
                    # print 'here-here'
                    return ys[-1]
                elif interp_type == 'zero':
                    return 0.0
            else:
                # sss = interpolator(x)
                # print('I am here', type(np.float(sss)), sss.size, sss.shape)
                return np.float(interpolator(x))

        if isinstance(z, (np.ndarray, list, tuple)):
            return np.array(map(pointwise, z), dtype=np.float)
        else:
            return pointwise(z)

    return interp
