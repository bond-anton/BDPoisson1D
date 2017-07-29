from __future__ import division, print_function
import numpy as np

from BDPoisson1D._helpers import fd_d2_matrix, interp_fn, points_for_refinement
from BDMesh import Mesh1DUniform

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

    def test_interp_fn(self):
        x = np.arange(10)
        k = 2
        y = k * x
        f = interp_fn(x, y, extrapolation='linear')
        self.assertEqual(f(10), 20)
        self.assertEqual(f(-10), -20)
        np.testing.assert_allclose(f(np.arange(20)), k * np.arange(20))
        f = interp_fn(x, y, extrapolation='last')
        self.assertEqual(f(10), x[-1] * k)
        self.assertEqual(f(-10),x[0] * k)
        np.testing.assert_allclose(f(np.arange(20) - 5), np.hstack((np.ones(5) * y[0],
                                                                    k * (np.arange(10)),
                                                                    np.ones(5) * y[-1])))
        f = interp_fn(x, y, extrapolation='zero')
        self.assertEqual(f(10), 0.0)
        self.assertEqual(f(-10), 0.0)
        np.testing.assert_allclose(f(np.arange(20) - 5), np.hstack((np.zeros(5),
                                                                    k * (np.arange(10)),
                                                                    np.zeros(5))))
        f = interp_fn(x, y, extrapolation='l')
        np.testing.assert_equal(f(10), np.array(np.nan))
        np.testing.assert_equal(f(-10), np.array(np.nan))
        np.testing.assert_allclose(f(np.arange(20) - 5), np.hstack((np.array([np.nan] * 5),
                                                                    k * (np.arange(10)),
                                                                    np.array([np.nan] * 5))))

    def test_points_for_refinement(self):
        mesh = Mesh1DUniform(0, 10, physical_step=1.0)
        threshold = 0.5
        indices = points_for_refinement(mesh, threshold=threshold)
        np.testing.assert_equal(indices[0], np.array([]))
        mesh.residual = [0.6, 0, 0, 0.6, 0.6, 0.5, 0, 0, 0.6, 0.6, 0.6]
        indices = points_for_refinement(mesh, threshold=threshold)
        np.testing.assert_equal(indices[0], np.array([0]))
        np.testing.assert_equal(indices[1], np.array([3, 4]))
        np.testing.assert_equal(indices[2], np.array([8, 9, 10]))
        with self.assertRaises(AssertionError):
            points_for_refinement(1, threshold=threshold)
        with self.assertRaises(AssertionError):
            points_for_refinement(mesh, threshold='a')
