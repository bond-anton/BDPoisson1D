from __future__ import division, print_function
import warnings
import numpy as np

from BDPoisson1D.NeumannLinear import neumann_poisson_solver_arrays, neumann_poisson_solver

import unittest


class TestNeumann(unittest.TestCase):

    def setUp(self):
        pass

    @staticmethod
    def y(x):
        if isinstance(x, (np.ndarray, list, tuple)):
            x = np.array(x)
        else:
            x = np.array([x])
        return -10 * np.sin(np.pi * x ** 2) / (2 * np.pi) + 3 * x ** 2 + x + 5

    def dy_numeric(self, x):
        """
        Numeric value of first derivative of y(x).
        :param x: 1D array of nodes.
        :return: y(x) first derivative values at x nodes.
        """
        return np.gradient(self.y(x), x, edge_order=2)

    def d2y_numeric(self, x):
        """
        Numeric calculation of second derivative of y(x) at given nodes.
        :param x: 1D array of nodes.
        :return: y(x) second derivative values at x nodes.
        """
        return np.gradient(self.dy_numeric(x), x, edge_order=2)

    def test_neumann_poisson_solver_arrays(self):
        start = 0.0
        stop = 2.0
        nodes = np.linspace(start, stop, num=51, endpoint=True)
        bc1 = self.dy_numeric(nodes)[0]
        bc2 = self.dy_numeric(nodes)[-1]
        _, residual1 = neumann_poisson_solver_arrays(nodes, self.d2y_numeric(nodes), bc1, bc2, y0=self.y(start)[0])
        nodes = np.linspace(start, stop, num=101, endpoint=True)
        bc1 = self.dy_numeric(nodes)[0]
        bc2 = self.dy_numeric(nodes)[-1]
        _, residual2 = neumann_poisson_solver_arrays(nodes, self.d2y_numeric(nodes), bc1, bc2, y0=self.y(start)[0])
        self.assertTrue(max(abs(residual2)) < max(abs(residual1)))
        bc1 = self.dy_numeric(nodes)[0]
        bc2 = self.dy_numeric(nodes)[-1] + 0.2
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            _, residual2 = neumann_poisson_solver_arrays(nodes, self.d2y_numeric(nodes), bc1, bc2, y0=self.y(start)[0])
            if len(w) > 0:
                self.assertTrue(len(w) == 1)
                self.assertTrue(issubclass(w[-1].category, UserWarning))
                self.assertTrue('Not well-posed' in str(w[-1].message))

    def test_neumann_poisson_solver(self):
        start = 0.0
        stop = 2.0
        nodes = np.linspace(start, stop, num=51, endpoint=True)
        bc1 = self.dy_numeric(nodes)[0]
        bc2 = self.dy_numeric(nodes)[-1]
        _, residual_1 = neumann_poisson_solver(nodes, self.d2y_numeric, bc1, bc2, y0=self.y(start)[0])
        nodes = np.linspace(start, stop, num=101, endpoint=True)
        bc1 = self.dy_numeric(nodes)[0]
        bc2 = self.dy_numeric(nodes)[-1]
        _, residual_2 = neumann_poisson_solver(nodes, self.d2y_numeric, bc1, bc2, y0=self.y(start)[0])
        self.assertTrue(max(abs(residual_2)) < max(abs(residual_1)))
        bc1 = self.dy_numeric(nodes)[0]
        bc2 = self.dy_numeric(nodes)[-1] + 0.5
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            _, residual_1 = neumann_poisson_solver(nodes, self.d2y_numeric, bc1, bc2, y0=self.y(start)[0])
            if len(w) > 0:
                self.assertTrue(len(w) == 1)
                self.assertTrue(issubclass(w[-1].category, UserWarning))
                self.assertTrue('Not well-posed' in str(w[-1].message))
