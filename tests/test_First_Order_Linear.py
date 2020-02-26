import numpy as np

from BDMesh import Mesh1DUniform
from BDPoisson1D.FirstOrderLinear import dirichlet_first_order_solver_arrays, dirichlet_first_order_solver
# from BDPoisson1D.DirichletLinear import dirichlet_poisson_solver_mesh_arrays, dirichlet_poisson_solver_mesh
# from BDPoisson1D.DirichletLinear import dirichlet_poisson_solver_amr
from BDPoisson1D.Function import Function, NumericGradient

import unittest


class TestFunction(Function):
    def evaluate(self, x):
        return -10 * np.sin(np.pi * np.array(x) ** 2) / (2 * np.pi) + 3 * np.array(x) ** 2 + np.array(x) + 5


class MixFunction(Function):
    """
    Some known differentiable function
    """
    def evaluate(self, x):
        # return np.ones(x.shape[0], dtype=np.double)
        # return np.zeros(x.shape[0], dtype=np.double)
        return np.asarray(x)**2 * np.cos(np.asarray(x))


class FFunction(Function):

    def __init__(self, p, y, yd):


class TestDirichletFirstOrder(unittest.TestCase):

    def setUp(self):
        self.y = TestFunction()
        self.dy_numeric = NumericGradient(self.y)
        self.p = MixFunction()

    def test_dirichlet_first_order_solver_arrays(self):
        start = -1.0
        stop = 2.0

        nodes = np.linspace(start, stop, num=51, endpoint=True)  # generate nodes
        p_nodes = self.p.evaluate(nodes)
        f_nodes = self.dy_numeric.evaluate(nodes) + p_nodes * self.y.evaluate(nodes)
        bc1 = self.y.evaluate([start])[0]  # left Dirichlet boundary condition
        bc2 = self.y.evaluate([stop])[0]  # right Dirichlet boundary condition

        result_1 = np.asarray(dirichlet_first_order_solver_arrays(nodes, p_nodes, f_nodes,
                                                                  bc1, bc2, j=1))
        nodes = np.linspace(start, stop, num=501, endpoint=True)
        p_nodes = self.p.evaluate(nodes)
        f_nodes = self.dy_numeric.evaluate(nodes) + p_nodes * self.y.evaluate(nodes)
        result_2 = np.asarray(dirichlet_first_order_solver_arrays(nodes, p_nodes, f_nodes,
                                                                  bc1, bc2, j=1))
        self.assertTrue(np.mean(result_2[:, 1]) < np.mean(result_1[:, 1]))

    def test_dirichlet_first_order_solver(self):
        start = -1.0
        stop = 2.0

        nodes = np.linspace(start, stop, num=51, endpoint=True)  # generate nodes
        p_nodes = self.p.evaluate(nodes)
        f_nodes = self.dy_numeric.evaluate(nodes) + p_nodes * self.y.evaluate(nodes)
        bc1 = self.y.evaluate([start])[0]  # left Dirichlet boundary condition
        bc2 = self.y.evaluate([stop])[0]  # right Dirichlet boundary condition

        result_1 = np.asarray(dirichlet_first_order_solver(nodes, self.p, self.f_nodes,
                                                                  bc1, bc2, j=1))
        nodes = np.linspace(start, stop, num=501, endpoint=True)
        p_nodes = self.p.evaluate(nodes)
        f_nodes = self.dy_numeric.evaluate(nodes) + p_nodes * self.y.evaluate(nodes)
        result_2 = np.asarray(dirichlet_first_order_solver_arrays(nodes, p_nodes, f_nodes,
                                                                  bc1, bc2, j=1))
        self.assertTrue(np.mean(result_2[:, 1]) < np.mean(result_1[:, 1]))