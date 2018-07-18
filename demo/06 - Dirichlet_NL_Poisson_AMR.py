from __future__ import division, print_function
import numpy as np


from matplotlib import pyplot as plt

from BDPoisson1D import dirichlet_non_linear_poisson_solver_amr
from BDPoisson1D import Function, Functional


class TestFunction(Function):
    """
    Some known differentiable function
    """
    def evaluate(self, x):
        return np.exp(-np.asarray(x) * 3)


class TestFunctional(Functional):
    def __init__(self, Nd, kT, f):
        super(TestFunctional, self).__init__(f)
        self.Nd = Nd
        self.kT = kT

    def evaluate(self, x):
        return self.Nd(np.asarray(x)) * (1 - (np.exp(-np.asarray(self.f.evaluate(x)) / self.kT)))


class TestFunctionalDf(Functional):
    def __init__(self, Nd, kT, f):
        super(TestFunctionalDf, self).__init__(f)
        self.Nd = Nd
        self.kT = kT

    def evaluate(self, x):
        return self.Nd(np.asarray(x)) / self.kT * np.exp(-np.asarray(self.f.evaluate(x)) / self.kT)


Nd = lambda x: np.ones_like(x)
kT = 1 / 20

Psi = TestFunction()
f = TestFunctional(Nd, kT, Psi)
dfdDPsi = TestFunctionalDf(Nd, kT, Psi)

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


start = 0.0
stop = 5.0
step = 0.5
bc1 = 1.0
bc2 = 0.0

Meshes = dirichlet_non_linear_poisson_solver_amr(start, stop, step, Psi, f, dfdDPsi, bc1, bc2,
                                                 max_iter=1000, residual_threshold=1.5e-3,
                                                 int_residual_threshold=1.5e-4,
                                                 max_level=20, mesh_refinement_threshold=1e-7)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)

flat_grid = Meshes.flatten()

for level in Meshes.levels:
    for mesh in Meshes.tree[level]:
        ax1.plot(mesh.physical_nodes, mesh.solution, colors[level] + '-')
        dPsi = np.gradient(mesh.solution, mesh.physical_nodes, edge_order=2)
        ax4.plot(mesh.physical_nodes, dPsi, colors[level] + '-')

ax2.plot(flat_grid.physical_nodes, flat_grid.residual, 'b-')
plot_tree(Meshes, ax3)
plt.show()
