from __future__ import division, print_function
import numpy as np
from matplotlib import pyplot as plt

from BDPoisson1D import dirichlet_poisson_solver_amr


def Y(x):
    # return x**2
    # return np.cos(2*np.pi*x)
    return np.exp(-abs(x))


# def f(x):
#    return -(2*np.pi)**2 * np.cos(2*np.pi * x)

def f(x):
    y = Y(x)
    dx = np.gradient(x)
    dy = np.gradient(y, dx, edge_order=2)
    d2y = np.gradient(dy, dx, edge_order=2)
    return d2y
    # return np.exp(-abs(x))


colors = ['b', 'g', 'y', 'k', 'm', 'c', 'b', 'g', 'y', 'k', 'm', 'c', 'b', 'g', 'y', 'k', 'm', 'c', 'b', 'g', 'y', 'k',
          'm', 'c']
nodes = np.linspace(0.0, 20.0, num=501, endpoint=True)
bc1 = Y(nodes[0])
bc2 = Y(nodes[-1])

meshes = dirichlet_poisson_solver_amr(nodes, f, bc1, bc2, 1.0e-5, max_level=20)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)

flat_grid, flat_sol, flat_res = meshes.flatten()

for level in meshes.levels:
    for mesh in meshes.Tree[level]:
        ax1.plot(mesh.phys_nodes(), mesh.solution, colors[level] + '-')
        # ax2.plot(mesh.phys_nodes(), mesh.residual, colors[level] + '-o')
        dx = np.gradient(mesh.phys_nodes())
        dPsi = np.gradient(mesh.solution, dx, edge_order=2)
        ax4.plot(mesh.phys_nodes(), dPsi, colors[level] + '-')

ax2.plot(flat_grid, flat_res, 'b-')
meshes.plot_tree(ax3)
# dx = np.gradient(flat_grid)
# dPsi = np.gradient(flat_sol, dx, edge_order=2)
# ax4.plot(flat_grid, dPsi, 'b-')
plt.show()
