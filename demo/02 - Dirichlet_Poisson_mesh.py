from __future__ import division, print_function
import numpy as np
from matplotlib import pyplot as plt

from BDPoisson1D import dirichlet_poisson_solver_mesh
from BDMesh import MeshUniform1D


def y(x):
    """
    Some known differentiable function
    :param x: 1D array of nodes
    :return: y(x) values of function at x nodes
    """
    return -10 * np.sin(np.pi * x**2) / (2 * np.pi) + 3 * x ** 2 + x + 5


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

start = -1.0
stop = 2.0
mesh = MeshUniform1D(start, stop, 0.02, y(start), y(stop), crop=None)

mesh = dirichlet_poisson_solver_mesh(mesh, f)  # solve Poisson equation

dy_solution = np.gradient(mesh.solution, mesh.physical_nodes, edge_order=2)
d2y_solution = np.gradient(dy_solution, mesh.physical_nodes, edge_order=2)

# Plot the result
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
ax1.plot(mesh.physical_nodes, f(mesh.physical_nodes), 'r-', label='f(x)')
ax1.plot(mesh.physical_nodes, d2y_solution, 'b-', label='d2y/dx2 (solution)')
ax1.legend()

ax2.plot(mesh.physical_nodes, dy_numeric(mesh.physical_nodes), 'r-', label='dy/dx')
ax2.plot(mesh.physical_nodes, dy_solution, 'b-', label='dy/dx (solution)')
ax2.legend()

ax3.plot(mesh.physical_nodes, mesh.solution, 'b-', label='solution')
ax3.plot(mesh.physical_nodes, y(mesh.physical_nodes), 'r-', label='y(x)')
ax3.legend()

ax4.plot(mesh.physical_nodes, mesh.residual, 'g-o', label='residual')
ax4.legend()
plt.show()
