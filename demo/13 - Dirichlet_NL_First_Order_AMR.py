import numpy as np
from matplotlib import pyplot as plt

from BDPoisson1D.FirstOrderNonLinear import dirichlet_non_linear_first_order_solver_amr
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


colors = ['b', 'g', 'y', 'k', 'm', 'c', 'b', 'g', 'y', 'k', 'm', 'c', 'b', 'g', 'y',
          'k', 'm', 'c', 'b', 'g', 'y', 'k', 'm', 'c']


def plot_tree(mesh_tree, axes=None):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b', 'c', 'm', 'y', 'k',
              'r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b', 'c', 'm', 'y', 'k']
    styles = ['-', ':', '--', '-', ':', '--', '-', '-', ':', '--', '-', ':', '--', '-',
              '-', ':', '--', '-', ':', '--', '-', '-', ':', '--', '-', ':', '--', '-',
              '-', ':', '--', '-', ':', '--', '-', '-', ':', '--', '-', ':', '--', '-',
              '-', ':', '--', '-', ':', '--', '-', '-', ':', '--', '-', ':', '--', '-']
    if axes is None:
        _, axes = plt.subplots()
    for level in mesh_tree.levels:
        for i, mesh in enumerate(mesh_tree.tree[level]):
            axes.plot(mesh.physical_nodes, np.ones(mesh.num) * level, colors[level] + styles[i] + 'x')
    axes.set_ylim([-1, max(mesh_tree.tree.keys()) + 1])
    axes.grid()

y0 = TestFunction()
dy0_numeric = NumericGradient(y0)
shift = np.pi * 11 + 1
start = -3*np.pi/2 + shift
stop = 3*np.pi/2 + shift + 0.5
step = 0.1

bc1 = y0.evaluate([start])[0]
bc2 = y0.evaluate([stop])[0]

y = GuessFunction()
p = MixFunction()

f = TestFunctional(y)
df_dy = TestFunctionalDf(y)

Meshes = dirichlet_non_linear_first_order_solver_amr(start, stop, step, y, p, f, df_dy, bc1, bc2, w=0.7,
                                                     max_iter=100, residual_threshold=1.5e-8,
                                                     int_residual_threshold=1e-13,
                                                     max_level=10, mesh_refinement_threshold=1e-7)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)

flat_grid = Meshes.flatten()

for level in Meshes.levels:
    for mesh in Meshes.tree[level]:
        ax1.plot(mesh.physical_nodes, mesh.solution, colors[level] + '-')
        try:
            dPsi = np.gradient(mesh.solution, mesh.physical_nodes, edge_order=2)
            ax4.plot(mesh.physical_nodes, dPsi, colors[level] + '-')
        except ValueError:
            pass

ax2.plot(flat_grid.physical_nodes, flat_grid.residual, 'b-')
plot_tree(Meshes, ax3)
plt.show()
