import math as m
import numpy as np

from BDMesh import Mesh1DUniform
from BDFunction1D import Function
from BDFunction1D.Differentiation import NumericGradient
from BDPoisson1D.DirichletLinear import dirichlet_poisson_solver_arrays, dirichlet_poisson_solver
from BDPoisson1D.DirichletLinear import dirichlet_poisson_solver_mesh_arrays, dirichlet_poisson_solver_mesh
from BDPoisson1D.DirichletLinear import dirichlet_poisson_solver_amr


import unittest


class TestFunction(Function):
    def evaluate_point(self, x):
        return -10 * m.sin(m.pi * x ** 2) / (2 * m.pi) + 3 * x ** 2 + x + 5


class TestDirichlet(unittest.TestCase):

    def setUp(self):
        self.y = TestFunction()
        self.dy_numeric = NumericGradient(self.y)
        self.d2y_numeric = NumericGradient(self.dy_numeric)

    def test_dirichlet_poisson_solver_arrays(self):
        start = -1.0
        stop = 2.0

        nodes = np.linspace(start, stop, num=51, endpoint=True)  # generate nodes
        bc1 = self.y.evaluate_point(start)  # left Dirichlet boundary condition
        bc2 = self.y.evaluate_point(stop)  # right Dirichlet boundary condition

        result_1 = np.asarray(dirichlet_poisson_solver_arrays(nodes, self.d2y_numeric.evaluate(nodes), bc1, bc2, j=1))
        nodes = np.linspace(start, stop, num=101, endpoint=True)
        result_2 = np.asarray(dirichlet_poisson_solver_arrays(nodes, self.d2y_numeric.evaluate(nodes), bc1, bc2, j=1))
        self.assertTrue(np.mean(result_2[5:-5, 1]) < np.mean(result_1[5:-5, 1]))

    def test_dirichlet_poisson_solver(self):
        start = -1.0
        stop = 2.0

        nodes = np.linspace(start, stop, num=51, endpoint=True)  # generate nodes
        bc1 = self.y.evaluate_point(start)  # left Dirichlet boundary condition
        bc2 = self.y.evaluate_point(stop)  # right Dirichlet boundary condition

        result_1 = np.asarray(dirichlet_poisson_solver(nodes, self.d2y_numeric, bc1, bc2, j=1.0).error(nodes))
        nodes = np.linspace(start, stop, num=101, endpoint=True)
        result_2 = np.asarray(dirichlet_poisson_solver(nodes, self.d2y_numeric, bc1, bc2, j=1.0).error(nodes))
        self.assertTrue(np.mean(result_2) < np.mean(result_1))

    def test_dirichlet_poisson_solver_mesh(self):
        start = -1.0
        stop = 2.0
        mesh_1 = Mesh1DUniform(start, stop,
                               boundary_condition_1=self.y.evaluate_point(start),
                               boundary_condition_2=self.y.evaluate_point(stop),
                               physical_step=0.02)
        dirichlet_poisson_solver_mesh_arrays(mesh_1, self.d2y_numeric.evaluate(mesh_1.physical_nodes))
        mesh_2 = Mesh1DUniform(start, stop,
                               boundary_condition_1=self.y.evaluate_point(start),
                               boundary_condition_2=self.y.evaluate_point(stop),
                               physical_step=0.01)
        dirichlet_poisson_solver_mesh_arrays(mesh_2, self.d2y_numeric.evaluate(mesh_2.physical_nodes))
        self.assertTrue(np.mean(np.asarray(mesh_2.residual)) < np.mean(np.asarray(mesh_1.residual)))

        mesh_1 = Mesh1DUniform(start, stop,
                               boundary_condition_1=self.y.evaluate_point(start),
                               boundary_condition_2=self.y.evaluate_point(stop),
                               physical_step=0.02)
        dirichlet_poisson_solver_mesh(mesh_1, self.d2y_numeric)
        mesh_2 = Mesh1DUniform(start, stop,
                               boundary_condition_1=self.y.evaluate_point(start),
                               boundary_condition_2=self.y.evaluate_point(stop),
                               physical_step=0.01)
        dirichlet_poisson_solver_mesh(mesh_2, self.d2y_numeric)
        self.assertTrue(np.mean(np.asarray(mesh_2.residual)) < np.mean(np.asarray(mesh_1.residual)))

    def test_dirichlet_poisson_solver_amr(self):
        start = 0.2
        stop = 1.2
        step = 0.01
        threshold = 1e-2
        max_level = 15
        y = dirichlet_poisson_solver_amr(start, stop, step, self.d2y_numeric,
                                         self.y.evaluate_point(start), self.y.evaluate_point(stop),
                                         threshold, max_level=max_level)
        self.assertTrue(max(abs(np.asarray(y.error(np.linspace(start, stop, num=int((stop - start)/step)))))) < 1.0e-2)
