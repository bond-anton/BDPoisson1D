from __future__ import division, print_function
import numpy as np


from matplotlib import pyplot as plt

from BDPoisson1D import dirichlet_non_linear_poisson_solver_amr

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


Nd = lambda x: np.ones_like(x)
kT = 1/40


def f(x, Psi):    
    return 2*(1 - (np.exp(-Psi(x)/kT)))


def dfdDPsi(x, Psi):
    return 2/kT * np.exp(-Psi(x)/kT)

Psi = lambda x: np.exp(-0.7*x)
start = 0.0
stop = 5
step = 0.5
bc1 = 1
bc2 = 0

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
