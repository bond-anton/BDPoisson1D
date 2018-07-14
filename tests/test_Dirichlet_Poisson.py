from __future__ import division, print_function
import warnings
import numpy as np

from BDMesh import Mesh1DUniform
from BDPoisson1D.DirichletLinear import dirichlet_poisson_solver_arrays, dirichlet_poisson_solver
from BDPoisson1D.DirichletLinear import dirichlet_poisson_solver_mesh_arrays, dirichlet_poisson_solver_mesh
from BDPoisson1D.DirichletLinear import dirichlet_poisson_solver_amr

import unittest


class TestDirichlet(unittest.TestCase):

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

    def test_dirichlet_poisson_solver_arrays(self):
        start = -1.0
        stop = 2.0

        nodes = np.linspace(start, stop, num=51, endpoint=True)  # generate nodes
        bc1 = self.y(start)[0]  # left Dirichlet boundary condition
        bc2 = self.y(stop)[0]  # right Dirichlet boundary condition

        _, residual_1 = dirichlet_poisson_solver_arrays(nodes, self.d2y_numeric(nodes), bc1, bc2, j=1)
        nodes = np.linspace(start, stop, num=101, endpoint=True)
        _, residual_2 = dirichlet_poisson_solver_arrays(nodes, self.d2y_numeric(nodes), bc1, bc2, j=1)
        self.assertTrue(max(abs(residual_2)) < max(abs(residual_1)))

    def test_dirichlet_poisson_solver(self):
        start = -1.0
        stop = 2.0

        nodes = np.linspace(start, stop, num=51, endpoint=True)  # generate nodes
        bc1 = self.y(start)[0]  # left Dirichlet boundary condition
        bc2 = self.y(stop)[0]  # right Dirichlet boundary condition

        _, residual_1 = dirichlet_poisson_solver(nodes, self.d2y_numeric, bc1, bc2, j=1)
        nodes = np.linspace(start, stop, num=101, endpoint=True)
        _, residual_2 = dirichlet_poisson_solver(nodes, self.d2y_numeric, bc1, bc2, j=1)
        self.assertTrue(max(abs(residual_2)) < max(abs(residual_1)))

    def test_dirichlet_poisson_solver_mesh(self):
        start = -1.0
        stop = 2.0
        mesh_1 = Mesh1DUniform(start, stop,
                               boundary_condition_1=self.y(start)[0], boundary_condition_2=self.y(stop)[0],
                               physical_step=0.02)
        mesh_1 = dirichlet_poisson_solver_mesh_arrays(mesh_1, self.d2y_numeric(mesh_1.physical_nodes))
        mesh_2 = Mesh1DUniform(start, stop,
                               boundary_condition_1=self.y(start)[0], boundary_condition_2=self.y(stop)[0],
                               physical_step=0.01)
        mesh_2 = dirichlet_poisson_solver_mesh_arrays(mesh_2, self.d2y_numeric(mesh_2.physical_nodes))
        self.assertTrue(max(abs(mesh_2.residual)) < max(abs(mesh_1.residual)))

        mesh_1 = Mesh1DUniform(start, stop,
                               boundary_condition_1=self.y(start)[0], boundary_condition_2=self.y(stop)[0],
                               physical_step=0.02)
        mesh_1 = dirichlet_poisson_solver_mesh(mesh_1, self.d2y_numeric)
        mesh_2 = Mesh1DUniform(start, stop,
                               boundary_condition_1=self.y(start)[0], boundary_condition_2=self.y(stop)[0],
                               physical_step=0.01)
        mesh_2 = dirichlet_poisson_solver_mesh(mesh_2, self.d2y_numeric)
        self.assertTrue(max(abs(mesh_2.residual)) < max(abs(mesh_1.residual)))

    def test_dirichlet_poisson_solver_amr(self):
        start = 0.2
        stop = 1.2
        step = 0.01
        threshold = 1e-2
        max_level = 15
        meshes = dirichlet_poisson_solver_amr(start, stop, step, self.d2y_numeric,
                                              self.y(start)[0], self.y(stop)[0], threshold, max_level=max_level)
        flat_mesh = meshes.flatten()
        if len(meshes.levels) < max_level:
            self.assertTrue(max(abs(flat_mesh.residual)) < 1.0e-2)
