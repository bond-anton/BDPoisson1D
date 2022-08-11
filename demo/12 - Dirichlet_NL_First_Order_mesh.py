import math as m
import numpy as np
from matplotlib import pyplot as plt

from BDPoisson1D.FirstOrderNonLinear import dirichlet_non_linear_first_order_solver_recurrent_mesh
from BDMesh import Mesh1DUniform
from BDFunction1D import Function
from BDFunction1D.Functional import Functional
from BDFunction1D.Differentiation import NumericGradient
from BDFunction1D.Interpolation import InterpolateFunction


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


y0 = TestFunction()
dy0_numeric = NumericGradient(y0)
shift = np.pi * 11 + 1
start = -3*np.pi/2 + shift
stop = 3*np.pi/2 + shift + 0.5

bc1 = y0.evaluate_point(start)
bc2 = y0.evaluate_point(stop)

y = GuessFunction()
p = MixFunction()

f = TestFunctional(y)
df_dy = TestFunctionalDf(y)

root_mesh = Mesh1DUniform(start, stop, bc1, bc2, 0.001)

dirichlet_non_linear_first_order_solver_recurrent_mesh(root_mesh, y, p, f, df_dy, w=0.0, max_iter=100, threshold=1e-7)
y = InterpolateFunction(root_mesh.physical_nodes, root_mesh.solution)

mesh_refinement_threshold = 1e-7
idxs = np.where(abs(np.asarray(root_mesh.residual)) > mesh_refinement_threshold)

dy = np.gradient(root_mesh.solution, root_mesh.physical_nodes, edge_order=2)

_, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
ax1.plot(np.asarray(root_mesh.physical_nodes), np.asarray(y0.evaluate(root_mesh.physical_nodes)), 'b-')
ax1.plot(np.asarray(root_mesh.physical_nodes), np.asarray(root_mesh.solution), 'r-')
ax1.plot(np.asarray(root_mesh.physical_nodes)[idxs], np.asarray(root_mesh.solution)[idxs], 'ro')
ax2.plot(np.asarray(root_mesh.physical_nodes), np.asarray(root_mesh.residual), 'r-')
ax2.plot(np.asarray(root_mesh.physical_nodes)[idxs], np.asarray(root_mesh.residual)[idxs], 'ro')
ax3.plot(np.asarray(root_mesh.physical_nodes), np.asarray(f.evaluate(root_mesh.physical_nodes)))
ax3.plot(np.asarray(root_mesh.physical_nodes), dy)
plt.show()
