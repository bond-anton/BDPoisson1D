from __future__ import division, print_function
import numpy as np

from BDMesh import Mesh1DUniform
from BDPoisson1D._helpers import interp_fn
from BDPoisson1D.DirichletNonLinear import dirichlet_non_linear_poisson_solver_arrays
from BDPoisson1D.DirichletNonLinear import dirichlet_non_linear_poisson_solver
from BDPoisson1D.DirichletNonLinear import dirichlet_non_linear_poisson_solver_mesh_arrays
from BDPoisson1D.DirichletNonLinear import dirichlet_non_linear_poisson_solver_mesh
from BDPoisson1D.DirichletNonLinear import dirichlet_non_linear_poisson_solver_recurrent_mesh
from BDPoisson1D.DirichletNonLinear import dirichlet_non_linear_poisson_solver_amr

import unittest


class TestDirichletNL(unittest.TestCase):

    def setUp(self):
        self.Nd = lambda x: np.ones_like(x)
        self.kT = 1 / 20
        self.bc1 = 1
        self.bc2 = 0

    def f(self, x, Psi):
        return self.Nd(x) * (1 - (np.exp(-Psi(x) / self.kT)))

    def dfdDPsi(self, x, Psi):
        return self.Nd(x) / self.kT * np.exp(-Psi(x) / self.kT)

    def test_dirichlet_poisson_solver_arrays(self):
        start = 0.0
        stop = 4.0
        nodes = np.linspace(start, stop, num=51, endpoint=True, dtype=np.float)
        Psi = lambda x: np.exp(-x * 3)
        R_1 = R_2 = 0
        for i in range(100):
            Psi_nodes, _, R_1 = dirichlet_non_linear_poisson_solver_arrays(nodes, Psi(nodes),
                                                                           self.f(nodes, Psi), self.dfdDPsi(nodes, Psi),
                                                                           bc1=1, bc2=0, j=1, w=1, rel=False)
            Psi = interp_fn(nodes, Psi_nodes)
        nodes = np.linspace(start, stop, num=101, endpoint=True, dtype=np.float)
        Psi = lambda x: np.exp(-x * 3)
        for i in range(100):
            Psi_nodes, _, R_2 = dirichlet_non_linear_poisson_solver_arrays(nodes, Psi(nodes),
                                                                           self.f(nodes, Psi), self.dfdDPsi(nodes, Psi),
                                                                           bc1=1, bc2=0, j=1, w=1, rel=False)
            Psi = interp_fn(nodes, Psi_nodes)
        self.assertTrue(max(abs(R_2)) < max(abs(R_1)))
        nodes = np.linspace(start, stop, num=51, endpoint=True, dtype=np.float)
        Psi = lambda x: np.exp(-x * 3)
        R_1 = R_2 = 0
        for i in range(100):
            Psi_nodes, _, R_1 = dirichlet_non_linear_poisson_solver_arrays(nodes, Psi(nodes),
                                                                           self.f(nodes, Psi), self.dfdDPsi(nodes, Psi),
                                                                           bc1=1, bc2=0, j=1, w=1, rel=True)
            Psi = interp_fn(nodes, Psi_nodes)
        nodes = np.linspace(start, stop, num=101, endpoint=True, dtype=np.float)
        Psi = lambda x: np.exp(-x * 3)
        for i in range(100):
            Psi_nodes, _, R_2 = dirichlet_non_linear_poisson_solver_arrays(nodes, Psi(nodes),
                                                                           self.f(nodes, Psi), self.dfdDPsi(nodes, Psi),
                                                                           bc1=1, bc2=0, j=1, w=1, rel=True)
            Psi = interp_fn(nodes, Psi_nodes)
        self.assertTrue(max(abs(R_2)) < max(abs(R_1)))
        nodes = np.linspace(start, stop, num=101, endpoint=True, dtype=np.float)
        Psi = lambda x: np.exp(-x * 3)
        with self.assertRaises(ZeroDivisionError):
            _, _, r = dirichlet_non_linear_poisson_solver_arrays(nodes, Psi(nodes),
                                                                 np.zeros(len(nodes)), self.dfdDPsi(nodes, Psi),
                                                                 bc1=1, bc2=0, j=1, w=1, rel=True)

    def test_dirichlet_poisson_solver(self):
        start = 0.0
        stop = 4.0
        nodes = np.linspace(start, stop, num=51, endpoint=True, dtype=np.float)
        Psi = lambda x: np.exp(-x * 3)
        R_1 = R_2 = 0
        for i in range(100):
            Psi, _, R_1 = dirichlet_non_linear_poisson_solver(nodes, Psi, self.f, self.dfdDPsi,
                                                              bc1=1, bc2=0, j=1, w=1)
        nodes = np.linspace(start, stop, num=101, endpoint=True, dtype=np.float)
        Psi = lambda x: np.exp(-x * 3)
        for i in range(100):
            Psi, _, R_2 = dirichlet_non_linear_poisson_solver(nodes, Psi, self.f, self.dfdDPsi,
                                                              bc1=1, bc2=0, j=1, w=1)
        self.assertTrue(max(abs(R_2)) < max(abs(R_1)))

    def test_dirichlet_poisson_solver_mesh_arays(self):
        start = 0.0
        stop = 4.0
        step = 0.5
        mesh_1 = Mesh1DUniform(start, stop, boundary_condition_1=1, boundary_condition_2=0,
                               physical_step=step, crop=None)
        Psi = lambda x: np.exp(-x * 3)
        for i in range(100):
            mesh_1, _ = dirichlet_non_linear_poisson_solver_mesh_arrays(mesh_1, Psi(mesh_1.physical_nodes),
                                                                        self.f(mesh_1.physical_nodes, Psi),
                                                                        self.dfdDPsi(mesh_1.physical_nodes, Psi), w=1)
            Psi = interp_fn(mesh_1.physical_nodes, mesh_1.solution)
        step = 0.1
        mesh_2 = Mesh1DUniform(start, stop, boundary_condition_1=1, boundary_condition_2=0,
                               physical_step=step, crop=None)
        Psi = lambda x: np.exp(-x * 3)
        for i in range(100):
            mesh_2, _ = dirichlet_non_linear_poisson_solver_mesh_arrays(mesh_2, Psi(mesh_2.physical_nodes),
                                                                        self.f(mesh_2.physical_nodes, Psi),
                                                                        self.dfdDPsi(mesh_2.physical_nodes, Psi), w=1)
            Psi = interp_fn(mesh_2.physical_nodes, mesh_2.solution)
        self.assertTrue(max(abs(mesh_2.residual)) < max(abs(mesh_1.residual)))

    def test_dirichlet_poisson_solver_mesh(self):
        start = 0.0
        stop = 4.0
        step = 0.5
        mesh_1 = Mesh1DUniform(start, stop, boundary_condition_1=1, boundary_condition_2=0,
                               physical_step=step, crop=None)
        Psi = lambda x: np.exp(-x * 3)
        for i in range(100):
            mesh_1, Psi, _ = dirichlet_non_linear_poisson_solver_mesh(mesh_1, Psi, self.f, self.dfdDPsi, w=1)
        step = 0.1
        mesh_2 = Mesh1DUniform(start, stop, boundary_condition_1=1, boundary_condition_2=0,
                               physical_step=step, crop=None)
        Psi = lambda x: np.exp(-x * 3)
        for i in range(100):
            mesh_2, Psi, _ = dirichlet_non_linear_poisson_solver_mesh(mesh_2, Psi, self.f, self.dfdDPsi, w=1)
        self.assertTrue(max(abs(mesh_2.residual)) < max(abs(mesh_1.residual)))

    def test_dirichlet_poisson_solver_recurrent_mesh(self):
        start = 0.0
        stop = 4.0
        step = 0.5
        threshold = 1e-6
        max_iter = 1000
        mesh_1 = Mesh1DUniform(start, stop, boundary_condition_1=1, boundary_condition_2=0,
                               physical_step=step, crop=None)
        Psi = lambda x: np.exp(-x * 3)
        mesh_1, Psi = dirichlet_non_linear_poisson_solver_recurrent_mesh(mesh_1, Psi, self.f, self.dfdDPsi,
                                                                         max_iter=max_iter, threshold=threshold)
        self.assertTrue(mesh_1.integrational_residual < threshold)

    def test_dirichlet_poisson_solver_mesh_amr(self):
        Psi = lambda x: np.exp(-0.7 * x)
        start = 0.0
        stop = 5
        step = 0.5
        bc1 = 1
        bc2 = 0

        residual_threshold = 1.5e-3
        int_residual_threshold = 1.5e-4
        mesh_refinement_threshold = 1e-7
        max_iter = 1000
        max_level = 20

        Meshes = dirichlet_non_linear_poisson_solver_amr(start, stop, step, Psi, self.f, self.dfdDPsi, bc1, bc2,
                                                         max_iter=max_iter, residual_threshold=residual_threshold,
                                                         int_residual_threshold=int_residual_threshold,
                                                         max_level=max_level,
                                                         mesh_refinement_threshold=mesh_refinement_threshold)
        flat_grid = Meshes.flatten()
        if len(Meshes.levels) < max_level:
            self.assertTrue(flat_grid.integrational_residual < int_residual_threshold)
            self.assertTrue(max(abs(flat_grid.residual)) < residual_threshold)
