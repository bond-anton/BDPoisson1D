from __future__ import division, print_function
import numpy as np
from matplotlib import pyplot as plt

from BDPoisson1D import dirichlet_poisson_solver


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

nodes = np.linspace(start, stop, num=51, endpoint=True)  # generate nodes
bc1 = y(start)  # left Dirichlet boundary condition
bc2 = y(stop)  # right Dirichlet boundary condition

y_solution, residual = dirichlet_poisson_solver(nodes, f, bc1, bc2, j=1)  # solve Poisson equation

dy_solution = np.gradient(y_solution, nodes, edge_order=2)
d2y_solution = np.gradient(dy_solution, nodes, edge_order=2)

# Plot the result
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
ax1.plot(nodes, f(nodes), 'r-', label='f(x)')
ax1.plot(nodes, d2y_solution, 'b-', label='d2y/dx2 (solution)')
ax1.legend()

ax2.plot(nodes, dy_numeric(nodes), 'r-', label='dy/dx')
ax2.plot(nodes, dy_solution, 'b-', label='dy/dx (solution)')
ax2.legend()

ax3.plot(nodes, y_solution, 'b-', label='solution')
ax3.plot(nodes, y(nodes), 'r-', label='y(x)')
ax3.legend()

ax4.plot(nodes, residual, 'g-o', label='residual')
ax4.legend()
plt.show()
