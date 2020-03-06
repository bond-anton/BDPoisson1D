import numpy as np

from BDMesh import Mesh1DUniform
from BDPoisson1D.FirstOrderNonLinear import dirichlet_non_linear_first_order_solver_arrays
from BDPoisson1D.FirstOrderNonLinear import dirichlet_non_linear_first_order_solver
from BDPoisson1D.FirstOrderNonLinear import dirichlet_non_linear_first_order_solver_mesh_arrays
from BDPoisson1D.FirstOrderNonLinear import dirichlet_non_linear_first_order_solver_mesh
from BDPoisson1D.FirstOrderNonLinear import dirichlet_non_linear_first_order_solver_recurrent_mesh
from BDPoisson1D.Function import Function, Functional, NumericGradient, InterpolateFunction

import unittest


class TestFunction(Function):
    """
    Some known differentiable function
    """

    def evaluate(self, x):
        xx = np.asarray(x)
        return np.sin(xx) ** 2


class TestFunctional(Functional):
    """
    f(x, y), RHS of the ODE
    """

    def evaluate(self, x):
        xx = np.asarray(x)
        yy = np.asarray(self.f.evaluate(x))
        result = np.empty_like(yy)
        idc = np.where(yy >= 0.5)
        ids = np.where(yy < 0.5)
        result[ids] = 2 * np.sign(np.cos(xx[ids])) * np.sin(xx[ids]) * np.sqrt(1 - yy[ids])
        result[idc] = 2 * np.sign(np.sin(xx[idc])) * np.cos(xx[idc]) * np.sqrt(yy[idc])
        return result


class TestFunctionalDf(Functional):
    """
    df/dy(x, y)
    """

    def evaluate(self, x):
        xx = np.asarray(x)
        yy = np.asarray(self.f.evaluate(x))
        result = np.empty_like(yy)
        idc = np.where(yy >= 0.5)
        ids = np.where(yy < 0.5)
        result[ids] = -np.sign(np.cos(xx[ids])) * np.sin(xx[ids]) / np.sqrt(1 - yy[ids])
        result[idc] = np.sign(np.sin(xx[idc])) * np.cos(xx[idc]) / np.sqrt(yy[idc])
        return result


class MixFunction(Function):
    """
    Some known differentiable function
    """

    def evaluate(self, x):
        return np.zeros(x.shape[0], dtype=np.double)


class GuessFunction(Function):
    """
    Some known differentiable function
    """

    def evaluate(self, x):
        return np.zeros(x.shape[0], dtype=np.double)


class TestDirichletFirstOrderNL(unittest.TestCase):

    def setUp(self):
        self.y0 = TestFunction()
        self.dy0_numeric = NumericGradient(self.y0)
        self.p = MixFunction()

    def test_dirichlet_first_order_solver_arrays(self):
        shift = np.pi * 11 + 1
        start = -3 * np.pi / 2 + shift
        stop = 3 * np.pi / 2 + shift + 0.5
        bc1 = self.y0.evaluate([start])[0]
        bc2 = self.y0.evaluate([stop])[0]

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
        bc1 = self.y0.evaluate([start])[0]
        bc2 = self.y0.evaluate([stop])[0]

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
        bc1 = self.y0.evaluate([start])[0]
        bc2 = self.y0.evaluate([stop])[0]

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
        bc1 = self.y0.evaluate([start])[0]
        bc2 = self.y0.evaluate([stop])[0]

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
        bc1 = self.y0.evaluate([start])[0]
        bc2 = self.y0.evaluate([stop])[0]

        y = GuessFunction()
        p = MixFunction()
        f = TestFunctional(y)
        df_dy = TestFunctionalDf(y)
        root_mesh = Mesh1DUniform(start, stop, bc1, bc2, num=1001)
        dirichlet_non_linear_first_order_solver_recurrent_mesh(root_mesh, y, p, f, df_dy, w=0.0, max_iter=100,
                                                               threshold=1e-7)
        self.assertTrue(np.sqrt(np.square(root_mesh.residual).mean()) < 1e-7)
