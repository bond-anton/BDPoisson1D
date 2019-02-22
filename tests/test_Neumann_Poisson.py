import warnings
import numpy as np

from BDMesh import Mesh1DUniform
from BDPoisson1D.NeumannLinear import neumann_poisson_solver_arrays, neumann_poisson_solver
from BDPoisson1D.NeumannLinear import neumann_poisson_solver_mesh, neumann_poisson_solver_mesh_arrays
from BDPoisson1D.NeumannLinear import neumann_poisson_solver_amr
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
        nodes = np.linspace(start, stop, num=5000, endpoint=True)
        bc1 = self.dy_numeric.evaluate(nodes)[0]
        bc2 = self.dy_numeric.evaluate(nodes)[-1]
        result_1 = np.asarray(neumann_poisson_solver_arrays(nodes, self.d2y_numeric.evaluate(nodes), bc1, bc2,
                                                            y0=self.y.evaluate(np.array([start]))[0]))
        nodes = np.linspace(start, stop, num=10000, endpoint=True)
        bc1 = self.dy_numeric.evaluate(nodes)[0]
        bc2 = self.dy_numeric.evaluate(nodes)[-1]
        result_2 = np.asarray(neumann_poisson_solver_arrays(nodes, self.d2y_numeric.evaluate(nodes), bc1, bc2,
                                                            y0=self.y.evaluate(np.array([start]))[0]))
        self.assertTrue(max(abs(result_2[5:-5, 1])) < max(abs(result_1[5:-5, 1])))
        bc1 = self.dy_numeric.evaluate(nodes)[0]
        bc2 = self.dy_numeric.evaluate(nodes)[-1] + 0.2
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            _ = neumann_poisson_solver_arrays(nodes, self.d2y_numeric.evaluate(nodes), bc1, bc2,
                                              y0=self.y.evaluate(np.array([start]))[0])
            if len(w) > 0:
                self.assertTrue(len(w) == 1)
                self.assertTrue(issubclass(w[-1].category, UserWarning))
                self.assertTrue('Not well-posed' in str(w[-1].message))

    def test_neumann_poisson_solver(self):
        start = 0.0
        stop = 2.0
        nodes = np.linspace(start, stop, num=5000, endpoint=True)
        bc1 = self.dy_numeric.evaluate(nodes)[0]
        bc2 = self.dy_numeric.evaluate(nodes)[-1]
        result_1 = np.asarray(neumann_poisson_solver(nodes, self.d2y_numeric, bc1, bc2,
                                                     y0=self.y.evaluate(np.array([start]))[0]))
        nodes = np.linspace(start, stop, num=10000, endpoint=True)
        bc1 = self.dy_numeric.evaluate(nodes)[0]
        bc2 = self.dy_numeric.evaluate(nodes)[-1]
        result_2 = np.asarray(neumann_poisson_solver(nodes, self.d2y_numeric, bc1, bc2,
                                                     y0=self.y.evaluate(np.array([start]))[0]))
        self.assertTrue(max(abs(result_2[5:-5, 1])) < max(abs(result_1[5:-5, 1])))
        bc1 = self.dy_numeric.evaluate(nodes)[0]
        bc2 = self.dy_numeric.evaluate(nodes)[-1] + 0.5
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            nodes = np.linspace(start, stop, num=100, endpoint=True)
            _ = neumann_poisson_solver(nodes, self.d2y_numeric, bc1, bc2,
                                       y0=self.y.evaluate(np.array([start]))[0])
            if len(w) > 0:
                self.assertTrue(len(w) == 1)
                self.assertTrue(issubclass(w[-1].category, UserWarning))
                self.assertTrue('Not well-posed' in str(w[-1].message))

    def test_neumann_poisson_solver_mesh(self):
        start = -1.0
        stop = 2.0
        mesh_1 = Mesh1DUniform(start, stop,
                               boundary_condition_1=self.y.evaluate([start])[0],
                               boundary_condition_2=self.y.evaluate([stop])[0],
                               physical_step=0.02)
        neumann_poisson_solver_mesh_arrays(mesh_1, self.d2y_numeric.evaluate(mesh_1.physical_nodes))
        mesh_2 = Mesh1DUniform(start, stop,
                               boundary_condition_1=self.y.evaluate([start])[0],
                               boundary_condition_2=self.y.evaluate([stop])[0],
                               physical_step=0.01)
        neumann_poisson_solver_mesh_arrays(mesh_2, self.d2y_numeric.evaluate(mesh_2.physical_nodes))
        self.assertTrue(max(abs(np.asarray(mesh_2.residual[10:-10]))) < max(abs(np.asarray(mesh_1.residual[10:-10]))))

        mesh_1 = Mesh1DUniform(start, stop,
                               boundary_condition_1=self.y.evaluate([start])[0],
                               boundary_condition_2=self.y.evaluate([stop])[0],
                               physical_step=0.02)
        neumann_poisson_solver_mesh(mesh_1, self.d2y_numeric)
        mesh_2 = Mesh1DUniform(start, stop,
                               boundary_condition_1=self.y.evaluate([start])[0],
                               boundary_condition_2=self.y.evaluate([stop])[0],
                               physical_step=0.01)
        neumann_poisson_solver_mesh(mesh_2, self.d2y_numeric)
        self.assertTrue(max(abs(np.asarray(mesh_2.residual[10:-10]))) < max(abs(np.asarray(mesh_1.residual[10:-10]))))

    def test_neumann_poisson_solver_amr(self):
        start = 0.2
        stop = 1.2
        step = 0.00001
        threshold = 1e-2
        max_level = 15
        meshes = neumann_poisson_solver_amr(start, stop, step, self.d2y_numeric,
                                            self.y.evaluate([start])[0], self.y.evaluate([stop])[0],
                                            threshold, max_level=max_level)
        flat_mesh = meshes.flatten()
        if len(meshes.levels) < max_level:
            self.assertTrue(max(abs(np.asarray(flat_mesh.residual[10:-10]))) < 1.0e-2)
