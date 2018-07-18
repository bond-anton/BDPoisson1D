from __future__ import division, print_function
import numpy as np

from BDPoisson1D._helpers import fd_d2_matrix, points_for_refinement, adjust_range
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

    def test_points_for_refinement(self):
        mesh = Mesh1DUniform(0, 10, physical_step=1.0)
        threshold = 0.5
        indices = points_for_refinement(mesh, threshold=threshold)
        np.testing.assert_equal(indices, np.array([]))
        mesh.residual = np.array([0.6, 0, 0, 0.6, 0.6, 0.5, 0, 0, 0.6, 0.6, 0.6])
        indices = points_for_refinement(mesh, threshold=threshold)
        np.testing.assert_equal(indices[0], np.array([0]))
        np.testing.assert_equal(indices[1], np.array([3, 4]))
        np.testing.assert_equal(indices[2], np.array([8, 9, 10]))
        with self.assertRaises(TypeError):
            points_for_refinement(1, threshold=threshold)
        with self.assertRaises(TypeError):
            points_for_refinement(mesh, threshold='a')

    def test_adjust_range(self):
        idx_range = np.array([0, 1])
        idx1, idx2, mesh_crop = adjust_range(idx_range, max_index=10, step_scale=2)
        self.assertEqual(idx1, 0)
        self.assertEqual(idx2, 2)
        self.assertEqual(mesh_crop, [0, 0])
        idx_range = np.array([9, 10])
        idx1, idx2, mesh_crop = adjust_range(idx_range, max_index=10, step_scale=2)
        self.assertEqual(idx1, 8)
        self.assertEqual(idx2, 10)
        self.assertEqual(mesh_crop, [0, 0])
        idx_range = np.array([4, 5])
        idx1, idx2, mesh_crop = adjust_range(idx_range, max_index=10, step_scale=2)
        self.assertEqual(idx1, 3)
        self.assertEqual(idx2, 6)
        self.assertEqual(mesh_crop, [0, 0])
        idx_range = np.array([4, 7])
        idx1, idx2, mesh_crop = adjust_range(idx_range, max_index=10, crop=[3, 3], step_scale=2)
        self.assertEqual(idx1, 2)
        self.assertEqual(idx2, 9)
        self.assertEqual(mesh_crop, [4, 4])
        idx_range = np.array([1, 9])
        idx1, idx2, mesh_crop = adjust_range(idx_range, max_index=10, crop=[3, 3], step_scale=2)
        self.assertEqual(idx1, 0)
        self.assertEqual(idx2, 10)
        self.assertEqual(mesh_crop, [2, 2])