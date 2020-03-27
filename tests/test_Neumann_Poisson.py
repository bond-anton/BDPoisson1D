import warnings
import math as m
import numpy as np

from BDMesh import Mesh1DUniform
from BDFunction1D import Function
from BDFunction1D.Differentiation import NumericGradient

from BDPoisson1D.NeumannLinear import neumann_poisson_solver_arrays, neumann_poisson_solver
from BDPoisson1D.NeumannLinear import neumann_poisson_solver_mesh, neumann_poisson_solver_mesh_arrays
from BDPoisson1D.NeumannLinear import neumann_poisson_solver_amr

import unittest


class TestFunction(Function):
    def evaluate_point(self, x):
        return -10 * m.sin(m.pi * x ** 2) / (2 * np.pi) + 3 * x ** 2 + x + 5


class TestNeumann(unittest.TestCase):

    def setUp(self):
        self.y = TestFunction()
        self.dy_numeric = NumericGradient(self.y)
        self.d2y_numeric = NumericGradient(self.dy_numeric)

    def test_neumann_poisson_solver_arrays(self):
        start = 0.0
        stop = 2.0
        nodes = np.linspace(start, stop, num=5000, endpoint=True)
        bc1 = self.dy_numeric.evaluate_point(start)
        bc2 = self.dy_numeric.evaluate_point(stop)
        result_1 = np.asarray(neumann_poisson_solver_arrays(nodes, self.d2y_numeric.evaluate(nodes), bc1, bc2,
                                                            y0=self.y.evaluate_point(start)))
        nodes = np.linspace(start, stop, num=10000, endpoint=True)
        bc1 = self.dy_numeric.evaluate_point(start)
        bc2 = self.dy_numeric.evaluate_point(stop)
        result_2 = np.asarray(neumann_poisson_solver_arrays(nodes, self.d2y_numeric.evaluate(nodes), bc1, bc2,
                                                            y0=self.y.evaluate_point(start)))
        self.assertTrue(max(abs(result_2[5:-5, 1])) < max(abs(result_1[5:-5, 1])))
        bc1 = self.dy_numeric.evaluate_point(start)
        bc2 = self.dy_numeric.evaluate_point(stop)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            _ = neumann_poisson_solver_arrays(nodes, self.d2y_numeric.evaluate(nodes), bc1, bc2,
                                              y0=self.y.evaluate_point(start))
            if len(w) > 0:
                self.assertTrue(len(w) == 1)
                self.assertTrue(issubclass(w[-1].category, UserWarning))
                self.assertTrue('Not well-posed' in str(w[-1].message))

    def test_neumann_poisson_solver(self):
        start = 0.0
        stop = 2.0
        nodes = np.linspace(start, stop, num=5000, endpoint=True)
        bc1 = self.dy_numeric.evaluate_point(start)
        bc2 = self.dy_numeric.evaluate_point(stop)
        result_1 = np.asarray(neumann_poisson_solver(nodes, self.d2y_numeric, bc1, bc2,
                                                     y0=self.y.evaluate_point(start)).error(nodes))
        nodes = np.linspace(start, stop, num=10000, endpoint=True)
        bc1 = self.dy_numeric.evaluate_point(start)
        bc2 = self.dy_numeric.evaluate_point(stop)
        result_2 = np.asarray(neumann_poisson_solver(nodes, self.d2y_numeric, bc1, bc2,
                                                     y0=self.y.evaluate_point(start)).error(nodes))
        self.assertTrue(max(abs(result_2[5:-5])) < max(abs(result_1[5:-5])))
        bc1 = self.dy_numeric.evaluate_point(start)
        bc2 = self.dy_numeric.evaluate_point(stop) + 0.5
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            nodes = np.linspace(start, stop, num=100, endpoint=True)
            _ = neumann_poisson_solver(nodes, self.d2y_numeric, bc1, bc2,
                                       y0=self.y.evaluate_point(start))
            if len(w) > 0:
                self.assertTrue(len(w) == 1)
                self.assertTrue(issubclass(w[-1].category, UserWarning))
                self.assertTrue('Not well-posed' in str(w[-1].message))

    def test_neumann_poisson_solver_mesh(self):
        start = -1.0
        stop = 2.0
        bc1 = 0.0
        bc2 = 3.0
        mesh_1 = Mesh1DUniform(start, stop, physical_step=0.02)
        mesh_1.boundary_condition_1 = bc1
        mesh_1.boundary_condition_2 = bc2
        neumann_poisson_solver_mesh_arrays(mesh_1, self.d2y_numeric.evaluate(mesh_1.physical_nodes))
        mesh_2 = Mesh1DUniform(start, stop, physical_step=0.01)
        mesh_2.boundary_condition_1 = bc1
        mesh_2.boundary_condition_2 = bc2
        neumann_poisson_solver_mesh_arrays(mesh_2, self.d2y_numeric.evaluate(mesh_2.physical_nodes))
        self.assertTrue(max(abs(np.asarray(mesh_2.residual[10:-10]))) < max(abs(np.asarray(mesh_1.residual[10:-10]))))

        mesh_1 = Mesh1DUniform(start, stop, physical_step=0.02)
        mesh_1.boundary_condition_1 = bc1
        mesh_1.boundary_condition_2 = bc2
        neumann_poisson_solver_mesh(mesh_1, self.d2y_numeric)
        mesh_2 = Mesh1DUniform(start, stop, physical_step=0.01)
        mesh_2.boundary_condition_1 = bc1
        mesh_2.boundary_condition_2 = bc2
        neumann_poisson_solver_mesh(mesh_2, self.d2y_numeric)
        self.assertTrue(max(abs(np.asarray(mesh_2.residual[10:-10]))) < max(abs(np.asarray(mesh_1.residual[10:-10]))))

    def test_neumann_poisson_solver_amr(self):
        start = 0.2
        stop = 2.0
        nodes = np.linspace(start, stop, num=5001, endpoint=True)
        bc1 = self.dy_numeric.evaluate_point(start)
        bc2 = self.dy_numeric.evaluate_point(stop)
        step = 0.00001
        threshold = 1e-2
        max_level = 15

        sol = neumann_poisson_solver_amr(start, stop, step, self.d2y_numeric,
                                         bc1, bc2, self.y.evaluate_point(start),
                                         threshold, max_level=max_level)
        self.assertTrue(max(abs(np.asarray(sol.error(sol.x)[10:-10]))) < 1.0e-2)
