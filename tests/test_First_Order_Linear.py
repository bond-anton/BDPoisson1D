import math as m
import numpy as np

from BDMesh import Mesh1DUniform
from BDFunction1D import Function
from BDFunction1D.Differentiation import NumericGradient

from BDPoisson1D.FirstOrderLinear import dirichlet_first_order_solver_arrays, dirichlet_first_order_solver
from BDPoisson1D.FirstOrderLinear import dirichlet_first_order_solver_mesh_arrays, dirichlet_first_order_solver_mesh

from BDPoisson1D.FirstOrderLinear import cauchy_first_order_solver_arrays, cauchy_first_order_solver
from BDPoisson1D.FirstOrderLinear import cauchy_first_order_solver_mesh_arrays, cauchy_first_order_solver_mesh

import unittest


class TestFunction(Function):
    def evaluate_point(self, x):
        return -10 * m.sin(m.pi * x ** 2) / (2 * m.pi) + 3 * x ** 2 + x + 5


class MixFunction(Function):
    """
    Some known differentiable function
    """
    def evaluate_point(self, x):
        # return 1.0
        # return 0.0
        return x**2 * m.cos(x)


class FFunction(Function):

    def __init__(self, p, y, dy):
        super(FFunction, self).__init__()
        self.p = p
        self.y = y
        self.dy = dy

    def evaluate_point(self, x):
        return self.dy.evaluate_point(x) + self.p.evaluate_point(x) * self.y.evaluate_point(x)


class TestDirichletFirstOrder(unittest.TestCase):

    def setUp(self):
        self.y = TestFunction()
        self.dy_numeric = NumericGradient(self.y)
        self.p = MixFunction()
        self.f = FFunction(self.p, self.y, self.dy_numeric)

    def test_dirichlet_first_order_solver_arrays(self):
        start = -1.0
        stop = 2.0

        nodes = np.linspace(start, stop, num=51, endpoint=True)  # generate nodes
        p_nodes = np.asarray(self.p.evaluate(nodes))
        f_nodes = np.asarray(self.dy_numeric.evaluate(nodes)) + p_nodes * np.asarray(self.y.evaluate(nodes))
        bc1 = self.y.evaluate_point(start)  # left Dirichlet boundary condition
        bc2 = self.y.evaluate_point(stop)  # right Dirichlet boundary condition

        result_1 = np.asarray(dirichlet_first_order_solver_arrays(nodes, p_nodes, f_nodes,
                                                                  bc1, bc2, j=1))
        err1 = np.abs(np.square(result_1 - self.y.evaluate(nodes)).mean())
        nodes = np.linspace(start, stop, num=501, endpoint=True)
        p_nodes = np.asarray(self.p.evaluate(nodes))
        f_nodes = np.asarray(self.dy_numeric.evaluate(nodes)) + p_nodes * np.asarray(self.y.evaluate(nodes))
        result_2 = np.asarray(dirichlet_first_order_solver_arrays(nodes, p_nodes, f_nodes,
                                                                  bc1, bc2, j=1))
        err2 = np.abs(np.square(result_2 - self.y.evaluate(nodes)).mean())
        self.assertTrue(err1 > err2)

    def test_dirichlet_first_order_solver(self):
        start = -1.0
        stop = 2.0

        nodes = np.linspace(start, stop, num=51, endpoint=True)  # generate nodes
        bc1 = self.y.evaluate_point(start)  # left Dirichlet boundary condition
        bc2 = self.y.evaluate_point(stop)  # right Dirichlet boundary condition

        result_1 = np.asarray(dirichlet_first_order_solver(nodes, self.p, self.f,
                                                           bc1, bc2, j=1).evaluate(nodes))
        err1 = np.abs(np.square(result_1 - self.y.evaluate(nodes)).mean())
        nodes = np.linspace(start, stop, num=501, endpoint=True)
        result_2 = np.asarray(dirichlet_first_order_solver(nodes, self.p, self.f,
                                                           bc1, bc2, j=1).evaluate(nodes))
        err2 = np.abs(np.square(result_2 - self.y.evaluate(nodes)).mean())
        self.assertTrue(err1 > err2)

    def test_dirichlet_first_order_solver_mesh(self):
        start = -1.0
        stop = 2.0
        mesh_1 = Mesh1DUniform(start, stop,
                               boundary_condition_1=self.y.evaluate_point(start),
                               boundary_condition_2=self.y.evaluate_point(stop),
                               physical_step=0.02)
        dirichlet_first_order_solver_mesh_arrays(mesh_1, self.p.evaluate(mesh_1.physical_nodes),
                                                 self.f.evaluate(mesh_1.physical_nodes))
        err1 = np.abs(np.square(mesh_1.solution - np.asarray(self.y.evaluate(mesh_1.physical_nodes))).mean())

        mesh_2 = Mesh1DUniform(start, stop,
                               boundary_condition_1=self.y.evaluate_point(start),
                               boundary_condition_2=self.y.evaluate_point(stop),
                               physical_step=0.01)
        dirichlet_first_order_solver_mesh_arrays(mesh_2, self.p.evaluate(mesh_2.physical_nodes),
                                                 self.f.evaluate(mesh_2.physical_nodes))
        err2 = np.abs(np.square(mesh_2.solution - np.asarray(self.y.evaluate(mesh_2.physical_nodes))).mean())
        self.assertTrue(err1 > err2)

        mesh_1 = Mesh1DUniform(start, stop,
                               boundary_condition_1=self.y.evaluate_point(start),
                               boundary_condition_2=self.y.evaluate_point(stop),
                               physical_step=0.02)
        result_1 = np.asarray(dirichlet_first_order_solver_mesh(mesh_1, self.p, self.f).evaluate(mesh_1.physical_nodes))
        np.testing.assert_allclose(result_1, mesh_1.solution)
        err1 = np.abs(np.square(mesh_1.solution - np.asarray(self.y.evaluate(mesh_1.physical_nodes))).mean())
        mesh_2 = Mesh1DUniform(start, stop,
                               boundary_condition_1=self.y.evaluate_point(start),
                               boundary_condition_2=self.y.evaluate_point(stop),
                               physical_step=0.01)
        result_2 = np.asarray(dirichlet_first_order_solver_mesh(mesh_2, self.p, self.f).evaluate(mesh_2.physical_nodes))
        np.testing.assert_allclose(result_2, mesh_2.solution)
        err2 = np.abs(np.square(mesh_2.solution - np.asarray(self.y.evaluate(mesh_2.physical_nodes))).mean())
        self.assertTrue(err1 > err2)

    def test_cauchy_first_order_solver_arrays(self):
        start = -1.0
        stop = 2.0

        nodes = np.linspace(start, stop, num=51, endpoint=True)  # generate nodes
        f_nodes = self.dy_numeric.evaluate(nodes)

        idx = 0
        bc = self.y.evaluate_point(nodes[idx])  # Initial condition
        result_1 = np.asarray(cauchy_first_order_solver_arrays(nodes, f_nodes, bc, idx, j=1))

        idx = 31
        bc = self.y.evaluate_point(nodes[idx])  # Initial condition
        result_1_2 = np.asarray(cauchy_first_order_solver_arrays(nodes, f_nodes, bc, idx, j=1))
        np.testing.assert_allclose(result_1, result_1_2, atol=1e-2)

        err1 = np.abs(np.square(result_1 - self.y.evaluate(nodes)).mean())

        nodes = np.linspace(start, stop, num=5001, endpoint=True)
        f_nodes = self.dy_numeric.evaluate(nodes)

        idx = 0
        bc = self.y.evaluate_point(nodes[idx])  # Initial condition
        result_2 = np.asarray(cauchy_first_order_solver_arrays(nodes, f_nodes, bc, idx, j=1))

        idx = 31
        bc = self.y.evaluate_point(nodes[idx])  # Initial condition
        result_2_2 = np.asarray(cauchy_first_order_solver_arrays(nodes, f_nodes, bc, idx, j=1))
        np.testing.assert_allclose(result_2, result_2_2)

        err2 = np.abs(np.square(result_2 - self.y.evaluate(nodes)).mean())
        self.assertTrue(err1 > err2)

    def test_cauchy_first_order_solver(self):
        start = -1.0
        stop = 2.0

        nodes = np.linspace(start, stop, num=51, endpoint=True)  # generate nodes

        idx = 0
        bc = self.y.evaluate_point(nodes[idx])  # Initial condition
        result_1 = np.asarray(cauchy_first_order_solver(nodes, self.dy_numeric, bc, idx, j=1).evaluate(nodes))

        idx = 31
        bc = self.y.evaluate_point(nodes[idx])  # Initial condition
        result_1_2 = np.asarray(cauchy_first_order_solver(nodes, self.dy_numeric, bc, idx, j=1).evaluate(nodes))
        np.testing.assert_allclose(result_1, result_1_2, atol=1e-2)

        err1 = np.abs(np.square(result_1 - self.y.evaluate(nodes)).mean())

        nodes = np.linspace(start, stop, num=5001, endpoint=True)

        idx = 0
        bc = self.y.evaluate_point(nodes[idx])  # Initial condition
        result_2 = np.asarray(cauchy_first_order_solver(nodes, self.dy_numeric, bc, idx, j=1).evaluate(nodes))

        idx = 31
        bc = self.y.evaluate_point(nodes[idx])  # Initial condition
        result_2_2 = np.asarray(cauchy_first_order_solver(nodes, self.dy_numeric, bc, idx, j=1).evaluate(nodes))
        np.testing.assert_allclose(result_2, result_2_2)

        err2 = np.abs(np.square(result_2 - self.y.evaluate(nodes)).mean())
        self.assertTrue(err1 > err2)

    def test_cauchy_first_order_solver_mesh_arrays(self):
        start = -1.0
        stop = 2.0

        mesh_1 = Mesh1DUniform(start, stop,
                               boundary_condition_1=self.y.evaluate_point(start),
                               boundary_condition_2=self.y.evaluate_point(stop),
                               physical_step=0.02)

        f_nodes = np.asarray(self.dy_numeric.evaluate(mesh_1.physical_nodes))

        idx = 0
        bc = self.y.evaluate_point(mesh_1.physical_nodes[idx])  # Initial condition
        np.asarray(cauchy_first_order_solver_mesh_arrays(mesh_1, f_nodes, bc, idx))
        result_1 = np.asarray(mesh_1.solution)

        idx = 31
        bc = self.y.evaluate_point(mesh_1.physical_nodes[idx])  # Initial condition
        np.asarray(cauchy_first_order_solver_mesh_arrays(mesh_1, f_nodes, bc, idx))
        result_1_2 = np.asarray(mesh_1.solution)

        np.testing.assert_allclose(result_1, result_1_2, atol=1e-2)

        err1 = np.abs(np.square(result_1 - self.y.evaluate(mesh_1.physical_nodes)).mean())

        mesh_2 = Mesh1DUniform(start, stop,
                               boundary_condition_1=self.y.evaluate_point(start),
                               boundary_condition_2=self.y.evaluate_point(stop),
                               physical_step=0.01)

        f_nodes = np.asarray(self.dy_numeric.evaluate(mesh_2.physical_nodes))

        idx = 0
        bc = self.y.evaluate_point(mesh_2.physical_nodes[idx])  # Initial condition
        np.asarray(cauchy_first_order_solver_mesh_arrays(mesh_2, f_nodes, bc, idx))
        result_2 = np.asarray(mesh_2.solution)

        idx = 31
        bc = self.y.evaluate_point(mesh_2.physical_nodes[idx])  # Initial condition
        np.asarray(cauchy_first_order_solver_mesh_arrays(mesh_2, f_nodes, bc, idx))
        result_2_2 = np.asarray(mesh_2.solution)

        np.testing.assert_allclose(result_2, result_2_2, atol=1e-2)

        err2 = np.abs(np.square(result_2 - self.y.evaluate(mesh_2.physical_nodes)).mean())
        self.assertTrue(err1 > err2)

    def test_cauchy_first_order_solver_mesh(self):
        start = -1.0
        stop = 2.0

        mesh_1 = Mesh1DUniform(start, stop,
                               boundary_condition_1=self.y.evaluate_point(start),
                               boundary_condition_2=self.y.evaluate_point(stop),
                               physical_step=0.02)

        idx = 0
        bc = self.y.evaluate_point(mesh_1.physical_nodes[idx])  # Initial condition
        result_1 = np.asarray(cauchy_first_order_solver_mesh(
            mesh_1, self.dy_numeric, bc, idx).evaluate(mesh_1.physical_nodes))

        idx = 31
        bc = self.y.evaluate_point(mesh_1.physical_nodes[idx])  # Initial condition
        result_1_2 = np.asarray(cauchy_first_order_solver_mesh(
            mesh_1, self.dy_numeric, bc, idx).evaluate(mesh_1.physical_nodes))
        np.testing.assert_allclose(result_1, result_1_2, atol=1e-2)

        err1 = np.abs(np.square(result_1 - self.y.evaluate(mesh_1.physical_nodes)).mean())

        mesh_2 = Mesh1DUniform(start, stop,
                               boundary_condition_1=self.y.evaluate_point(start),
                               boundary_condition_2=self.y.evaluate_point(stop),
                               physical_step=0.01)

        idx = 0
        bc = self.y.evaluate_point(mesh_2.physical_nodes[idx])  # Initial condition
        result_2 = np.asarray(cauchy_first_order_solver_mesh(
            mesh_2, self.dy_numeric, bc, idx).evaluate(mesh_2.physical_nodes))

        idx = 31
        bc = self.y.evaluate_point(mesh_2.physical_nodes[idx])  # Initial condition
        result_2_2 = np.asarray(cauchy_first_order_solver_mesh(
            mesh_2, self.dy_numeric, bc, idx).evaluate(mesh_2.physical_nodes))
        np.testing.assert_allclose(result_2, result_2_2, atol=1e-2)

        err2 = np.abs(np.square(result_2 - self.y.evaluate(mesh_2.physical_nodes)).mean())
        self.assertTrue(err1 > err2)
