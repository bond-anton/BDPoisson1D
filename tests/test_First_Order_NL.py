import math as m
import numpy as np

from BDMesh import Mesh1DUniform
from BDFunction1D import Function
from BDFunction1D.Functional import Functional
from BDFunction1D.Interpolation import InterpolateFunction
from BDFunction1D.Differentiation import NumericGradient

from BDPoisson1D.FirstOrderNonLinear import dirichlet_non_linear_first_order_solver_arrays
from BDPoisson1D.FirstOrderNonLinear import dirichlet_non_linear_first_order_solver
from BDPoisson1D.FirstOrderNonLinear import dirichlet_non_linear_first_order_solver_mesh_arrays
from BDPoisson1D.FirstOrderNonLinear import dirichlet_non_linear_first_order_solver_mesh
from BDPoisson1D.FirstOrderNonLinear import dirichlet_non_linear_first_order_solver_recurrent_mesh

import unittest


class TestFunction(Function):
    """
    Some known differentiable function
    """

    def evaluate_point(self, x):
        return m.sin(x) ** 2


class TestFunctional(Functional):
    """
    f(x, y), RHS of the ODE
    """

    def evaluate_point(self, x):
        y = self.f.evaluate_point(x)
        if y >= 0.5:
            return 2 * np.sign(m.sin(x)) * m.cos(x) * m.sqrt(y)
        else:
            return 2 * np.sign(m.cos(x)) * m.sin(x) * m.sqrt(1 - y)


class TestFunctionalDf(Functional):
    """
    df/dy(x, y)
    """

    def evaluate_point(self, x):
        y = self.f.evaluate_point(x)
        if y >= 0.5:
            return np.sign(m.sin(x)) * m.cos(x) / m.sqrt(y)
        else:
            return -np.sign(m.cos(x)) * m.sin(x) / m.sqrt(1 - y)


class MixFunction(Function):
    """
    Some known differentiable function
    """

    def evaluate_point(self, x):
        return 0.0


class GuessFunction(Function):
    """
    Some known differentiable function
    """

    def evaluate_point(self, x):
        return 0.0


class TestDirichletFirstOrderNL(unittest.TestCase):

    def setUp(self):
        self.y0 = TestFunction()
        self.dy0_numeric = NumericGradient(self.y0)
        self.p = MixFunction()

    def test_dirichlet_first_order_solver_arrays(self):
        shift = np.pi * 11 + 1
        start = -3 * np.pi / 2 + shift
        stop = 3 * np.pi / 2 + shift + 0.5
        bc1 = self.y0.evaluate_point(start)
        bc2 = self.y0.evaluate_point(stop)

        y = GuessFunction()
        p = MixFunction()
        f = TestFunctional(y)
        df_dy = TestFunctionalDf(y)
        nodes = np.linspace(start, stop, num=1001, endpoint=True)  # generate nodes
        y_nodes = y.evaluate(nodes)
        p_nodes = p.evaluate(nodes)
        f_nodes = f.evaluate(nodes)
        df_dy_nodes = df_dy.evaluate(nodes)
        w = 1.0
        min_w = 0.3
        mse_threshold = 1e-15
        i = 0
        max_iterations = 100
        mse_old = 1e20
        while i < max_iterations:
            result = dirichlet_non_linear_first_order_solver_arrays(nodes, y_nodes, p_nodes,
                                                                    f_nodes, df_dy_nodes,
                                                                    bc1, bc2, j=1.0, w=w)
            y = InterpolateFunction(nodes, result[:, 0])

            f.f = y
            df_dy.f = y

            y_nodes = y.evaluate(nodes)
            p_nodes = p.evaluate(nodes)
            f_nodes = f.evaluate(nodes)
            df_dy_nodes = df_dy.evaluate(nodes)

            mse = np.sqrt(np.square(result[:, 1]).mean())
            if mse > mse_old:
                if w > min_w:
                    w -= 0.1
                    print(i, ' -> reduced W to', w)
                    continue
                else:
                    print('Not converging anymore. W =', w)
                    break
            if mse < mse_threshold:
                break

            mse_old = mse
            i += 1
        print('reached mse:', mse, 'in', i, 'iterations')
        self.assertTrue(mse < mse_threshold)

    def test_dirichlet_first_order_solver(self):
        shift = np.pi * 11 + 1
        start = -3 * np.pi / 2 + shift
        stop = 3 * np.pi / 2 + shift + 0.5
        bc1 = self.y0.evaluate_point(start)
        bc2 = self.y0.evaluate_point(stop)

        y = GuessFunction()
        p = MixFunction()
        f = TestFunctional(y)
        df_dy = TestFunctionalDf(y)
        nodes = np.linspace(start, stop, num=1001, endpoint=True)  # generate nodes
        w = 1.0
        min_w = 0.3
        mse_threshold = 1e-15
        i = 0
        max_iterations = 100
        mse_old = 1e20
        while i < max_iterations:
            result = dirichlet_non_linear_first_order_solver(nodes, y, p, f, df_dy, bc1, bc2, j=1.0, w=w)
            y = InterpolateFunction(nodes, result[:, 0])
            f.f = y
            df_dy.f = y

            mse = np.sqrt(np.square(result[:, 1]).mean())
            if mse > mse_old:
                if w > min_w:
                    w -= 0.1
                    print(i, ' -> reduced W to', w)
                    continue
                else:
                    print('Not converging anymore. W =', w)
                    break
            if mse < mse_threshold:
                break

            mse_old = mse
            i += 1
        print('reached mse:', mse, 'in', i, 'iterations')
        self.assertTrue(mse < mse_threshold)

    def test_dirichlet_first_order_solver_mesh_arrays(self):
        shift = np.pi * 11 + 1
        start = -3 * np.pi / 2 + shift
        stop = 3 * np.pi / 2 + shift + 0.5
        bc1 = self.y0.evaluate_point(start)
        bc2 = self.y0.evaluate_point(stop)

        y = GuessFunction()
        p = MixFunction()
        f = TestFunctional(y)
        df_dy = TestFunctionalDf(y)
        root_mesh = Mesh1DUniform(start, stop, bc1, bc2, num=1001)
        y_nodes = y.evaluate(root_mesh.physical_nodes)
        p_nodes = p.evaluate(root_mesh.physical_nodes)
        f_nodes = f.evaluate(root_mesh.physical_nodes)
        df_dy_nodes = df_dy.evaluate(root_mesh.physical_nodes)
        w = 1.0
        min_w = 0.3
        mse_threshold = 1e-15
        i = 0
        max_iterations = 100
        mse_old = 1e20
        while i < max_iterations:
            dirichlet_non_linear_first_order_solver_mesh_arrays(root_mesh, y_nodes, p_nodes,
                                                                f_nodes, df_dy_nodes, w=w)
            y = InterpolateFunction(root_mesh.physical_nodes, root_mesh.solution)

            f.f = y
            df_dy.f = y

            y_nodes = y.evaluate(root_mesh.physical_nodes)
            p_nodes = p.evaluate(root_mesh.physical_nodes)
            f_nodes = f.evaluate(root_mesh.physical_nodes)
            df_dy_nodes = df_dy.evaluate(root_mesh.physical_nodes)

            mse = np.sqrt(np.square(root_mesh.residual).mean())
            if mse > mse_old:
                if w > min_w:
                    w -= 0.1
                    print(i, ' -> reduced W to', w)
                    continue
                else:
                    print('Not converging anymore. W =', w)
                    break
            if mse < mse_threshold:
                break

            mse_old = mse
            i += 1
        print('reached mse:', mse, 'in', i, 'iterations')
        self.assertTrue(mse < mse_threshold)

    def test_dirichlet_first_order_solver_mesh(self):
        shift = np.pi * 11 + 1
        start = -3 * np.pi / 2 + shift
        stop = 3 * np.pi / 2 + shift + 0.5
        bc1 = self.y0.evaluate_point(start)
        bc2 = self.y0.evaluate_point(stop)

        y = GuessFunction()
        p = MixFunction()
        f = TestFunctional(y)
        df_dy = TestFunctionalDf(y)
        root_mesh = Mesh1DUniform(start, stop, bc1, bc2, num=1001)
        w = 1.0
        min_w = 0.3
        mse_threshold = 1e-15
        i = 0
        max_iterations = 100
        mse_old = 1e20
        while i < max_iterations:
            dirichlet_non_linear_first_order_solver_mesh(root_mesh, y, p, f, df_dy, w=w)
            y = InterpolateFunction(root_mesh.physical_nodes, root_mesh.solution)
            f.f = y
            df_dy.f = y

            mse = np.sqrt(np.square(root_mesh.residual).mean())
            if mse > mse_old:
                if w > min_w:
                    w -= 0.1
                    print(i, ' -> reduced W to', w)
                    continue
                else:
                    print('Not converging anymore. W =', w)
                    break
            if mse < mse_threshold:
                break

            mse_old = mse
            i += 1
        print('reached mse:', mse, 'in', i, 'iterations')
        self.assertTrue(mse < mse_threshold)

    def test_dirichlet_first_order_solver_recurrent_mesh(self):
        shift = np.pi * 11 + 1
        start = -3 * np.pi / 2 + shift
        stop = 3 * np.pi / 2 + shift + 0.5
        bc1 = self.y0.evaluate_point(start)
        bc2 = self.y0.evaluate_point(stop)

        y = GuessFunction()
        p = MixFunction()
        f = TestFunctional(y)
        df_dy = TestFunctionalDf(y)
        root_mesh = Mesh1DUniform(start, stop, bc1, bc2, num=1001)
        dirichlet_non_linear_first_order_solver_recurrent_mesh(root_mesh, y, p, f, df_dy, w=0.0, max_iter=100,
                                                               threshold=1e-7)
        self.assertTrue(np.sqrt(np.square(root_mesh.residual).mean()) < 1e-7)
