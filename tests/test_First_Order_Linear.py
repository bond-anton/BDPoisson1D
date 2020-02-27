import numpy as np

from BDMesh import Mesh1DUniform
from BDPoisson1D.FirstOrderLinear import dirichlet_first_order_solver_arrays, dirichlet_first_order_solver
from BDPoisson1D.FirstOrderLinear import dirichlet_first_order_solver_mesh_arrays, dirichlet_first_order_solver_mesh
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

    def __init__(self, p, y, dy):
        self.p = p
        self.y = y
        self.dy = dy

    def evaluate(self, x):
        return self.dy.evaluate(x) + self.p.evaluate(x) * self.y.evaluate(x)


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
        bc1 = self.y.evaluate([start])[0]  # left Dirichlet boundary condition
        bc2 = self.y.evaluate([stop])[0]  # right Dirichlet boundary condition

        result_1 = np.asarray(dirichlet_first_order_solver(nodes, self.p, self.f,
                                                           bc1, bc2, j=1))
        nodes = np.linspace(start, stop, num=501, endpoint=True)
        result_2 = np.asarray(dirichlet_first_order_solver(nodes, self.p, self.f,
                                                           bc1, bc2, j=1))
        self.assertTrue(np.mean(result_2[:, 1]) < np.mean(result_1[:, 1]))

    def test_dirichlet_first_order_solver_mesh(self):
        start = -1.0
        stop = 2.0
        mesh_1 = Mesh1DUniform(start, stop,
                               boundary_condition_1=self.y.evaluate([start])[0],
                               boundary_condition_2=self.y.evaluate([stop])[0],
                               physical_step=0.02)
        dirichlet_first_order_solver_mesh_arrays(mesh_1, self.p.evaluate(mesh_1.physical_nodes),
                                                 self.f.evaluate(mesh_1.physical_nodes))
        mesh_2 = Mesh1DUniform(start, stop,
                               boundary_condition_1=self.y.evaluate([start])[0],
                               boundary_condition_2=self.y.evaluate([stop])[0],
                               physical_step=0.01)
        dirichlet_first_order_solver_mesh_arrays(mesh_2, self.p.evaluate(mesh_2.physical_nodes),
                                                 self.f.evaluate(mesh_2.physical_nodes))
        self.assertTrue(np.mean(np.asarray(mesh_2.residual)) < np.mean(np.asarray(mesh_1.residual)))

        mesh_1 = Mesh1DUniform(start, stop,
                               boundary_condition_1=self.y.evaluate([start])[0],
                               boundary_condition_2=self.y.evaluate([stop])[0],
                               physical_step=0.02)
        dirichlet_first_order_solver_mesh(mesh_1, self.p, self.f)
        mesh_2 = Mesh1DUniform(start, stop,
                               boundary_condition_1=self.y.evaluate([start])[0],
                               boundary_condition_2=self.y.evaluate([stop])[0],
                               physical_step=0.01)
        dirichlet_first_order_solver_mesh(mesh_2, self.p, self.f)
        self.assertTrue(np.mean(np.asarray(mesh_2.residual)) < np.mean(np.asarray(mesh_1.residual)))
