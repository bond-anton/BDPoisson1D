from __future__ import division, print_function
import numpy as np
from matplotlib import pyplot as plt

from BDPoisson1D import dirichlet_poisson_solver_amr


def plot_tree(mesh_tree, ax=None):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b', 'c', 'm', 'y', 'k',
              'r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b', 'c', 'm', 'y', 'k']
    styles = ['-', ':', '--', '-', ':', '--', '-', '-', ':', '--', '-', ':', '--', '-',
              '-', ':', '--', '-', ':', '--', '-', '-', ':', '--', '-', ':', '--', '-',
              '-', ':', '--', '-', ':', '--', '-', '-', ':', '--', '-', ':', '--', '-',
              '-', ':', '--', '-', ':', '--', '-', '-', ':', '--', '-', ':', '--', '-']
    if ax is None:
        _, ax = plt.subplots()
    for level in mesh_tree.levels:
        for i, mesh in enumerate(mesh_tree.tree[level]):
            ax.plot(mesh.physical_nodes, np.ones(mesh.num) * level, colors[level] + styles[i] + 'x')
    ax.set_ylim([-1, max(mesh_tree.tree.keys()) + 1])
    ax.grid()

def y(x):
    """
    Some known differentiable function
    :param x: 1D array of nodes
    :return: y(x) values of function at x nodes
    """
    return -20 * np.sin(2 * np.pi * x) / (2 * np.pi) + 3 * x ** 2 + x + 5


def dy_numeric(x):
    """
    Numeric value of first derivative of y(x)
    :param x: 1D array of nodes
    :return: y(x) first derivative values at x nodes
    """
    return np.gradient(y(x), x, edge_order=2)


def d2y_numeric(x):
    """
    Numeric calculation of second derivative of y(x) at given nodes
    :param x: 1D array of nodes
    :return: y(x) second derivative values at x nodes
    """
    return np.gradient(dy_numeric(x), x, edge_order=2)


def f(x):
    return d2y_numeric(x)

start = 0.2
stop = 1.2

root_nodes = np.linspace(start, stop, num=101, endpoint=True)  # generate nodes
bc1 = y(start)  # left Dirichlet boundary condition
bc2 = y(stop)  # right Dirichlet boundary condition

meshes = dirichlet_poisson_solver_amr(root_nodes, f, bc1, bc2, 5.0e-3, max_level=10, verbose=True)
flat_mesh = meshes.flatten()

nodes = flat_mesh.physical_nodes
y_solution = flat_mesh.solution
residual = flat_mesh.residual

dy_solution = np.gradient(y_solution, nodes, edge_order=2)
d2y_solution = np.gradient(dy_solution, nodes, edge_order=2)

# Plot the result
colors = ['b', 'g', 'y', 'k', 'm', 'c', 'b', 'g', 'y', 'k', 'm', 'c', 'b', 'g', 'y',
          'k', 'm', 'c', 'b', 'g', 'y', 'k', 'm', 'c']

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True)
ax1.plot(nodes, f(nodes), 'r-', label='f(x)')
ax1.plot(nodes, d2y_solution, 'b-', label='d2y/dx2 (solution)')
ax2.plot(nodes, dy_numeric(nodes), 'r-', label='dy/dx')
ax2.plot(nodes, dy_solution, 'b-', label='dy/dx (solution)')
ax3.plot(nodes, y_solution, 'b-', label='solution')
ax3.plot(nodes, y(nodes), 'r-', label='y(x)')
ax4.plot(nodes, residual, 'g-o', label='residual')

'''
for level in meshes.levels:
    for mesh in meshes.tree[level]:
        dy_solution = np.gradient(mesh.solution, mesh.physical_nodes, edge_order=2)
        d2y_solution = np.gradient(dy_solution, mesh.physical_nodes, edge_order=2)
        ax1.plot(mesh.physical_nodes, d2y_solution, colors[level] + '-')
        ax2.plot(mesh.physical_nodes, dy_solution, colors[level] + '-o')
        ax3.plot(mesh.physical_nodes, mesh.solution, colors[level] + '-')
        ax4.plot(mesh.physical_nodes, mesh.residual, colors[level] + '-')
'''
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()

plot_tree(meshes, ax5)

plt.show()

