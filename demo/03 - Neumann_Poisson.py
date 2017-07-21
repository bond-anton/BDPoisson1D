from __future__ import division, print_function
import numpy as np
from matplotlib import pyplot as plt

from BDPoisson1D import neumann_poisson_solver


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


f = lambda x: d2y_numeric(x)

start = 0.0
stop = 2.0
nodes = np.linspace(start, stop, num=5001, endpoint=True)
bc1 = dy_numeric(nodes)[0]
bc2 = dy_numeric(nodes)[-1]
integral = np.trapz(f(nodes), nodes)
print(integral, bc2 - bc1)
print(np.allclose(integral, bc2 - bc1))

y_solution, residual = neumann_poisson_solver(nodes, f, bc1, bc2, y0=y(start), debug=True)
dy_solution = np.gradient(y_solution, nodes, edge_order=2)
d2y_solution = np.gradient(dy_solution, nodes, edge_order=2)

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
