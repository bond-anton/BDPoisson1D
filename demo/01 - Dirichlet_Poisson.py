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
    return np.cos(2 * np.pi * x)


def f_analytic(x):
    """
    Analytic value of second derivative of y(x)
    :param x: 1D array of nodes
    :return: y(x) second derivative values at x nodes
    """
    return -(2 * np.pi)**2 * np.cos(2 * np.pi * x)


def f_numeric(x):
    """
    Numeric calculation of second derivative of y(x) at given nodes
    :param x: 1D array of nodes
    :return: y(x) second derivative values at x nodes
    """
    dy = np.gradient(y(x), x, edge_order=2)
    d2y = np.gradient(dy, x, edge_order=2)
    return d2y

nodes = np.linspace(-1.0, 1.0, num=51, endpoint=True)  # generate nodes
bc1 = y(nodes[0])  # left Dirichlet boundary condition
bc2 = y(nodes[-1])  # right Dirichlet boundary condition

y_solution, residual = dirichlet_poisson_solver(nodes, f_numeric, bc1, bc2, j=1, debug=True)  # solve Poisson equation

# Plot the result
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(nodes, y(nodes), 'r-', label='y(x)')
ax1.plot(nodes, y_solution, 'b-', label='solution')
ax1.legend()

ax2.plot(nodes, residual, 'g-o', label='d2y/dx residual')
ax2.legend()
plt.show()
