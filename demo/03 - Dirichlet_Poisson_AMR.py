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
    #return -20 * np.sin(2 * np.pi / x**2) / (2 * np.pi) + 3 * x ** 2 + x + 5


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
step = 0.1

bc1 = y(start)  # left Dirichlet boundary condition
bc2 = y(stop)  # right Dirichlet boundary condition

meshes = dirichlet_poisson_solver_amr(start, stop, step, f, bc1, bc2, 1.0e-3, max_level=10, verbose=True)
flat_mesh = meshes.flatten()
y_solution = flat_mesh.solution
dy_solution = np.gradient(y_solution, flat_mesh.physical_nodes, edge_order=2)
d2y_solution = np.gradient(dy_solution, flat_mesh.physical_nodes, edge_order=2)

# Plot the result
fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True)
ax[0][0].plot(flat_mesh.physical_nodes, d2y_solution, 'b-', label='d2y/dx2 (solution)')
ax[0][0].plot(flat_mesh.physical_nodes, f(flat_mesh.physical_nodes), 'r-', label='f(x)')
ax[0][1].plot(flat_mesh.physical_nodes, y_solution, 'b-', label='solution')
ax[0][1].plot(flat_mesh.physical_nodes, y(flat_mesh.physical_nodes), 'r-', label='y(x)')
ax[1][0].plot(flat_mesh.physical_nodes, flat_mesh.residual, 'g-o', label='residual')
plot_tree(meshes, ax[1][1])
ax[0][0].legend()
ax[0][1].legend()
ax[1][0].legend()
plt.show()
