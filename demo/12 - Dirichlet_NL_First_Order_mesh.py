import numpy as np
from matplotlib import pyplot as plt

from BDPoisson1D.FirstOrderNonLinear import dirichlet_non_linear_first_order_solver_mesh
from BDPoisson1D.FirstOrderNonLinear import dirichlet_non_linear_first_order_solver_recurrent_mesh
from BDPoisson1D import Function, Functional, NumericGradient, InterpolateFunction
from BDMesh import Mesh1DUniform


class TestFunction(Function):
    """
    Some known differentiable function
    """
    def evaluate(self, x):
        xx = np.asarray(x)
        return np.sin(xx)**2


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


y0 = TestFunction()
dy0_numeric = NumericGradient(y0)
shift = np.pi * 11 + 1
start = -3*np.pi/2 + shift
stop = 3*np.pi/2 + shift + 0.5

bc1 = y0.evaluate([start])[0]
bc2 = y0.evaluate([stop])[0]

y = GuessFunction()
p = MixFunction()

f = TestFunctional(y)
df_dy = TestFunctionalDf(y)

root_mesh = Mesh1DUniform(start, stop, bc1, bc2, 0.001)

dirichlet_non_linear_first_order_solver_recurrent_mesh(root_mesh, y, p, f, df_dy, w=0.0, max_iter=100, threshold=1e-14)
y = InterpolateFunction(root_mesh.physical_nodes, root_mesh.solution)

mesh_refinement_threshold = 1e-7
idxs = np.where(abs(np.asarray(root_mesh.residual)) > mesh_refinement_threshold)

dy = np.gradient(root_mesh.solution, root_mesh.physical_nodes, edge_order=2)

_, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
ax1.plot(root_mesh.physical_nodes, y0.evaluate(root_mesh.physical_nodes), 'b-')
ax1.plot(root_mesh.physical_nodes, root_mesh.solution, 'r-')
ax1.plot(np.asarray(root_mesh.physical_nodes)[idxs], np.asarray(root_mesh.solution)[idxs], 'ro')
ax2.plot(root_mesh.physical_nodes, root_mesh.residual, 'r-')
ax2.plot(np.asarray(root_mesh.physical_nodes)[idxs], np.asarray(root_mesh.residual)[idxs], 'ro')
ax3.plot(root_mesh.physical_nodes, f.evaluate(root_mesh.physical_nodes))
ax3.plot(root_mesh.physical_nodes, dy)
plt.show()
