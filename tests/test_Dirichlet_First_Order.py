import numpy as np

from BDMesh import Mesh1DUniform
from BDPoisson1D.FirstOrderLinear import dirichlet_first_order_solver_arrays #, dirichlet_poisson_solver
# from BDPoisson1D.DirichletLinear import dirichlet_poisson_solver_mesh_arrays, dirichlet_poisson_solver_mesh
# from BDPoisson1D.DirichletLinear import dirichlet_poisson_solver_amr
from BDPoisson1D.Function import Function, NumericGradient

import unittest


class TestFunction(Function):
    def evaluate(self, x):
        return -10 * np.sin(np.pi * np.array(x) ** 2) / (2 * np.pi) + 3 * np.array(x) ** 2 + np.array(x) + 5


class TestDirichletFirstOrder(unittest.TestCase):

    def setUp(self):
        self.y = TestFunction()
        self.dy_numeric = NumericGradient(self.y)
        self.d2y_numeric = NumericGradient(self.dy_numeric)

    def test_dirichlet_poisson_solver_arrays(self):
        start = -1.0
        stop = 2.0

        nodes = np.linspace(start, stop, num=51, endpoint=True)  # generate nodes
        p_nodes = np.zeros(nodes.shape[0], dtype=np.double)
        bc1 = self.y.evaluate([start])[0]  # left Dirichlet boundary condition
        bc2 = self.y.evaluate([stop])[0]  # right Dirichlet boundary condition

        result_1 = np.asarray(dirichlet_first_order_solver_arrays(nodes, p_nodes, self.dy_numeric.evaluate(nodes),
                                                                  bc1, bc2, j=1))
        nodes = np.linspace(start, stop, num=101, endpoint=True)
        p_nodes = np.zeros(nodes.shape[0], dtype=np.double)
        result_2 = np.asarray(dirichlet_first_order_solver_arrays(nodes, p_nodes, self.dy_numeric.evaluate(nodes),
                                                                  bc1, bc2, j=1))
        self.assertTrue(max(abs(result_2[5:-5, 1])) < max(abs(result_1[5:-5, 1])))
