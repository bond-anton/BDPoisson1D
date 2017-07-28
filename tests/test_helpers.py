from __future__ import division, print_function
import numpy as np
from BDPoisson1D._helpers import fd_d2_matrix

import unittest


class TestHelpers(unittest.TestCase):

    def setUp(self):
        pass

    def test_fd_d2_matrix(self):
        m = fd_d2_matrix(2)
        np.testing.assert_equal(m.todense(), np.array([[-2, 1],
                                                       [1, -2]]))
        m = fd_d2_matrix(3)
        np.testing.assert_equal(m.todense(), np.array([[-2, 1, 0],
                                                       [1, -2, 1],
                                                       [0, 1, -2]]))
        with self.assertRaises(TypeError):
            fd_d2_matrix(3.5)
