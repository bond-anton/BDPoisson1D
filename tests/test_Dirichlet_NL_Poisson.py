from __future__ import division, print_function
import numpy as np

from BDMesh import Mesh1DUniform
from BDPoisson1D.DirichletNonLinear import dirichlet_non_linear_poisson_solver_arrays
from BDPoisson1D.DirichletNonLinear import dirichlet_non_linear_poisson_solver
from BDPoisson1D.DirichletNonLinear import dirichlet_non_linear_poisson_solver_mesh_arrays
from BDPoisson1D.DirichletNonLinear import dirichlet_non_linear_poisson_solver_mesh
from BDPoisson1D.DirichletNonLinear import dirichlet_non_linear_poisson_solver_recurrent_mesh
from BDPoisson1D.DirichletNonLinear import dirichlet_non_linear_poisson_solver_amr

from BDPoisson1D.Function import Function, Functional, InterpolateFunction

import unittest


class testPsi(Function):
    def evaluate(self, x):
        return np.exp(-np.asarray(x) * 3.0)


class testF(Functional):

    def __init__(self, Nd, kT, f):
        super(testF, self).__init__(f)
        self.__f = f
        self.Nd = Nd
        self.kT = kT

    def evaluate(self, x):
        return self.Nd * (1 - (np.exp(-np.asarray(self.__f.evaluate(np.asarray(x))) / self.kT)))


class testdFdPsi(Functional):

    def __init__(self, Nd, kT, f):
        super(testdFdPsi, self).__init__(f)
        self.__f = f
        self.Nd = Nd
        self.kT = kT

    def evaluate(self, x):
        return self.Nd / self.kT * np.exp(-np.asarray(self.__f.evaluate(np.asarray(x))) / self.kT)


class TestDirichletNL(unittest.TestCase):

    def setUp(self):
        self.Nd = 1.0
        self.kT = 0.05
        self.bc1 = 1.0
        self.bc2 = 0.0

    def test_dirichlet_poisson_solver_arrays(self):
        start = 0.0
        stop = 4.0
        R_1 = R_2 = 0
        nodes = np.linspace(start, stop, num=51, endpoint=True, dtype=np.float)
        Psi = testPsi()
        f = testF(self.Nd, self.kT, Psi)
        dfdDPsi = testdFdPsi(self.Nd, self.kT, Psi)
        for i in range(100):
            Psi_nodes, _, R_1 = dirichlet_non_linear_poisson_solver_arrays(nodes, Psi.evaluate(nodes),
                                                                           f.evaluate(nodes),
                                                                           dfdDPsi.evaluate(nodes),
                                                                           bc1=1, bc2=0, j=1, w=1)
            print(max(abs(R_1)))
            Psi = InterpolateFunction(nodes, Psi_nodes)
            f.__f = Psi
            dfdDPsi.__f = Psi
        nodes = np.linspace(start, stop, num=101, endpoint=True, dtype=np.float)
        Psi = testPsi()
        f.__f = Psi
        dfdDPsi.__f = Psi
        for i in range(100):
            Psi_nodes, _, R_2 = dirichlet_non_linear_poisson_solver_arrays(nodes, Psi.evaluate(nodes),
                                                                           f.evaluate(nodes),
                                                                           dfdDPsi.evaluate(nodes),
                                                                           bc1=1, bc2=0, j=1, w=1)
            Psi = InterpolateFunction(nodes, Psi_nodes)
            f.__f = Psi
            dfdDPsi.__f = Psi
        self.assertTrue(max(abs(R_2)) < max(abs(R_1)))

"""
    def test_dirichlet_poisson_solver(self):
        start = 0.0
        stop = 4.0
        R_1 = R_2 = 0
        nodes = np.linspace(start, stop, num=51, endpoint=True, dtype=np.float)
        self.Psi = testPsi()
        self.f = testF(self.Nd, self.kT, self.Psi)
        self.dfdDPsi = testdFdPsi(self.Nd, self.kT, self.Psi)
        for i in range(100):
            self.Psi, _, R_1 = dirichlet_non_linear_poisson_solver(nodes, self.Psi, self.f, self.dfdDPsi,
                                                                   bc1=1, bc2=0, j=1, w=1)
            self.f = testF(self.Nd, self.kT, self.Psi)
            self.dfdDPsi = testdFdPsi(self.Nd, self.kT, self.Psi)
        nodes = np.linspace(start, stop, num=101, endpoint=True, dtype=np.float)
        for i in range(100):
            self.Psi, _, R_2 = dirichlet_non_linear_poisson_solver(nodes, self.Psi, self.f, self.dfdDPsi,
                                                                   bc1=1, bc2=0, j=1, w=1)
            self.f = testF(self.Nd, self.kT, self.Psi)
            self.dfdDPsi = testdFdPsi(self.Nd, self.kT, self.Psi)
        self.assertTrue(max(abs(R_2)) < max(abs(R_1)))

    def test_dirichlet_poisson_solver_mesh_arays(self):
        start = 0.0
        stop = 4.0
        step = 0.5
        mesh_1 = Mesh1DUniform(start, stop, boundary_condition_1=1, boundary_condition_2=0, physical_step=step)
        for i in range(100):
            mesh_1, _ = dirichlet_non_linear_poisson_solver_mesh_arrays(mesh_1,
                                                                        self.Psi.evaluate(mesh_1.physical_nodes),
                                                                        self.f.evaluate(mesh_1.physical_nodes),
                                                                        self.dfdDPsi.evaluate(mesh_1.physical_nodes),
                                                                        w=1)
            self.Psi = InterpolateFunction(mesh_1.physical_nodes, mesh_1.solution)
            self.f = testF(self.Nd, self.kT, self.Psi)
            self.dfdDPsi = testdFdPsi(self.Nd, self.kT, self.Psi)
        step = 0.1
        mesh_2 = Mesh1DUniform(start, stop, boundary_condition_1=1, boundary_condition_2=0, physical_step=step)
        for i in range(100):
            mesh_2, _ = dirichlet_non_linear_poisson_solver_mesh_arrays(mesh_2,
                                                                        self.Psi.evaluate(mesh_2.physical_nodes),
                                                                        self.f.evaluate(mesh_2.physical_nodes),
                                                                        self.dfdDPsi.evaluate(mesh_2.physical_nodes),
                                                                        w=1)
            self.Psi = InterpolateFunction(mesh_2.physical_nodes, mesh_2.solution)
            self.f = testF(self.Nd, self.kT, self.Psi)
            self.dfdDPsi = testdFdPsi(self.Nd, self.kT, self.Psi)
        self.assertTrue(max(abs(mesh_2.residual)) < max(abs(mesh_1.residual)))

    def test_dirichlet_poisson_solver_mesh(self):
        start = 0.0
        stop = 4.0
        step = 0.5
        mesh_1 = Mesh1DUniform(start, stop, boundary_condition_1=1, boundary_condition_2=0, physical_step=step)
        self.Psi = testPsi()
        self.f = testF(self.Nd, self.kT, self.Psi)
        self.dfdDPsi = testdFdPsi(self.Nd, self.kT, self.Psi)
        for i in range(100):
            mesh_1, self.Psi, _ = dirichlet_non_linear_poisson_solver_mesh(mesh_1, self.Psi, self.f, self.dfdDPsi, w=1)
            self.f = testF(self.Nd, self.kT, self.Psi)
            self.dfdDPsi = testdFdPsi(self.Nd, self.kT, self.Psi)
        step = 0.1
        mesh_2 = Mesh1DUniform(start, stop, boundary_condition_1=1, boundary_condition_2=0, physical_step=step)
        self.Psi = testPsi()
        self.f = testF(self.Nd, self.kT, self.Psi)
        self.dfdDPsi = testdFdPsi(self.Nd, self.kT, self.Psi)
        for i in range(100):
            mesh_2, self.Psi, _ = dirichlet_non_linear_poisson_solver_mesh(mesh_2, self.Psi, self.f, self.dfdDPsi, w=1)
            self.f = testF(self.Nd, self.kT, self.Psi)
            self.dfdDPsi = testdFdPsi(self.Nd, self.kT, self.Psi)
        self.assertTrue(max(abs(mesh_2.residual)) < max(abs(mesh_1.residual)))

    def test_dirichlet_poisson_solver_recurrent_mesh(self):
        start = 0.0
        stop = 4.0
        step = 0.5
        threshold = 1e-6
        max_iter = 1000
        mesh_1 = Mesh1DUniform(start, stop, boundary_condition_1=1, boundary_condition_2=0, physical_step=step)
        self.Psi = testPsi()
        self.f = testF(self.Nd, self.kT, self.Psi)
        self.dfdDPsi = testdFdPsi(self.Nd, self.kT, self.Psi)
        mesh_1, self.Psi = dirichlet_non_linear_poisson_solver_recurrent_mesh(mesh_1, self.Psi, self.f, self.dfdDPsi,
                                                                              max_iter=max_iter, threshold=threshold)
        self.assertTrue(mesh_1.integrational_residual < threshold)

    def test_dirichlet_poisson_solver_mesh_amr(self):
        self.Psi = testPsi()
        self.f = testF(self.Nd, self.kT, self.Psi)
        self.dfdDPsi = testdFdPsi(self.Nd, self.kT, self.Psi)
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

        Meshes = dirichlet_non_linear_poisson_solver_amr(start, stop, step, self.Psi, self.f, self.dfdDPsi, bc1, bc2,
                                                         max_iter=max_iter, residual_threshold=residual_threshold,
                                                         int_residual_threshold=int_residual_threshold,
                                                         max_level=max_level,
                                                         mesh_refinement_threshold=mesh_refinement_threshold)
        flat_grid = Meshes.flatten()
        if len(Meshes.levels) < max_level:
            self.assertTrue(flat_grid.integrational_residual < int_residual_threshold)
            self.assertTrue(max(abs(flat_grid.residual)) < residual_threshold)

        # self.Psi = testPsi()
        self.f = testF(self.Nd, self.kT, self.Psi)
        self.dfdDPsi = testdFdPsi(self.Nd, self.kT, self.Psi)

        residual_threshold = 1.5e-6
        int_residual_threshold = 1.5e-4
        mesh_refinement_threshold = 1e-5
        max_iter = 1000
        max_level = 20
        Meshes = dirichlet_non_linear_poisson_solver_amr(start, stop, step, self.Psi, self.f, self.dfdDPsi, bc1, bc2,
                                                         max_iter=max_iter, residual_threshold=residual_threshold,
                                                         int_residual_threshold=int_residual_threshold,
                                                         max_level=max_level,
                                                         mesh_refinement_threshold=mesh_refinement_threshold)
        flat_grid = Meshes.flatten()
        if len(Meshes.levels) < max_level:
            self.assertTrue(flat_grid.integrational_residual < int_residual_threshold)
"""