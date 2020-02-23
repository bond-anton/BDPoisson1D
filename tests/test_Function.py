import numpy as np
from scipy.interpolate import interp1d

from BDPoisson1D.Function import Function, InterpolateFunction, Functional, NumericGradient

import unittest


class TestFunction(unittest.TestCase):

    def setUp(self):
        pass

    def test_Function(self):
        f = Function()
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), x)

    def test_new_Function(self):
        class test_F(Function):
            def evaluate(self, x):
                return np.sqrt(x)
        f = test_F()
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), np.sqrt(x))

    def test_interpolate_Function(self):
        x = np.linspace(0.0, 2 * np.pi, num=101, dtype=np.float)
        y = np.sin(x)
        f = InterpolateFunction(x, y)
        x_new = np.linspace(0.0, 2 * np.pi, num=201, dtype=np.float)
        f1 = interp1d(x, y, kind='linear')
        np.testing.assert_allclose(f.evaluate(x_new), f1(x_new), atol=1e-12)

    def test_Functional(self):
        class test_F1(Function):
            def evaluate(self, x):
                return np.sqrt(x)

        class test_F2(Function):
            def evaluate(self, x):
                return np.sin(x)

        class test_Functional(Functional):
            def __init__(self, f):
                super(test_Functional, self).__init__(f)

        f = test_Functional(test_F1())
        x = np.arange(100, dtype=np.float)
        np.testing.assert_allclose(f.evaluate(x), np.sqrt(x))
        f.f = test_F2()
        np.testing.assert_allclose(f.evaluate(x), np.sin(x))

    def test_numeric_diff(self):

        class test_F(Function):
            def evaluate(self, x):
                return np.sin(x)

        f = test_F()
        df = NumericGradient(f)
        x = np.linspace(0.0, 2 * np.pi, num=501, dtype=np.float)
        np.testing.assert_allclose(df.evaluate(x), np.cos(x), atol=1e-4)

        y = np.sin(x)
        f = InterpolateFunction(x, y)
        df = NumericGradient(f)
        np.testing.assert_allclose(df.evaluate(x), np.cos(x), atol=1e-4)
