from __future__ import division, print_function
import warnings
import numpy as np

from BDPoisson1D.NeumannLinear import neumann_poisson_solver_arrays, neumann_poisson_solver
from BDPoisson1D.Function import Function, NumericDiff

import unittest


class TestFunction(Function):
    def evaluate(self, x):
        return -10 * np.sin(np.pi * np.array(x) ** 2) / (2 * np.pi) + 3 * np.array(x) ** 2 + np.array(x) + 5


class TestNeumann(unittest.TestCase):

    def setUp(self):
        self.y = TestFunction()
        self.dy_numeric = NumericDiff(self.y)
        self.d2y_numeric = NumericDiff(self.dy_numeric)

    def test_neumann_poisson_solver_arrays(self):
        start = 0.0
        stop = 2.0
        nodes = np.linspace(start, stop, num=51, endpoint=True)
        bc1 = self.dy_numeric.evaluate(nodes)[0]
        bc2 = self.dy_numeric.evaluate(nodes)[-1]
        _, residual1 = neumann_poisson_solver_arrays(nodes, self.d2y_numeric.evaluate(nodes), bc1, bc2,
                                                     y0=self.y.evaluate(np.array([start]))[0])
        nodes = np.linspace(start, stop, num=101, endpoint=True)
        bc1 = self.dy_numeric.evaluate(nodes)[0]
        bc2 = self.dy_numeric.evaluate(nodes)[-1]
        _, residual2 = neumann_poisson_solver_arrays(nodes, self.d2y_numeric.evaluate(nodes), bc1, bc2,
                                                     y0=self.y.evaluate(np.array([start]))[0])
        self.assertTrue(max(abs(residual2)) < max(abs(residual1)))
        bc1 = self.dy_numeric.evaluate(nodes)[0]
        bc2 = self.dy_numeric.evaluate(nodes)[-1] + 0.2
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            _, residual2 = neumann_poisson_solver_arrays(nodes, self.d2y_numeric.evaluate(nodes), bc1, bc2,
                                                         y0=self.y.evaluate(np.array([start]))[0])
            if len(w) > 0:
                self.assertTrue(len(w) == 1)
                self.assertTrue(issubclass(w[-1].category, UserWarning))
                self.assertTrue('Not well-posed' in str(w[-1].message))

    def test_neumann_poisson_solver(self):
        start = 0.0
        stop = 2.0
        nodes = np.linspace(start, stop, num=51, endpoint=True)
        bc1 = self.dy_numeric.evaluate(nodes)[0]
        bc2 = self.dy_numeric.evaluate(nodes)[-1]
        _, residual_1 = neumann_poisson_solver(nodes, self.d2y_numeric, bc1, bc2,
                                               y0=self.y.evaluate(np.array([start]))[0])
        nodes = np.linspace(start, stop, num=101, endpoint=True)
        bc1 = self.dy_numeric.evaluate(nodes)[0]
        bc2 = self.dy_numeric.evaluate(nodes)[-1]
        _, residual_2 = neumann_poisson_solver(nodes, self.d2y_numeric, bc1, bc2,
                                               y0=self.y.evaluate(np.array([start]))[0])
        self.assertTrue(max(abs(residual_2)) < max(abs(residual_1)))
        bc1 = self.dy_numeric.evaluate(nodes)[0]
        bc2 = self.dy_numeric.evaluate(nodes)[-1] + 0.5
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            _, residual_1 = neumann_poisson_solver(nodes, self.d2y_numeric, bc1, bc2,
                                                   y0=self.y.evaluate(np.array([start]))[0])
            if len(w) > 0:
                self.assertTrue(len(w) == 1)
                self.assertTrue(issubclass(w[-1].category, UserWarning))
                self.assertTrue('Not well-posed' in str(w[-1].message))
