from __future__ import division, print_function
import numpy as np
from matplotlib import pyplot as plt

from BDPoisson1D import dirichlet_poisson_solver
from BDPoisson1D import Function, NumericDiff


class TestFunction(Function):
    """
    Some known differentiable function
    """
    def evaluate(self, x):
        return -10 * np.sin(np.pi * np.asarray(x)**2) / (2 * np.pi) + 3 * np.asarray(x) ** 2 + np.asarray(x) + 5

y = TestFunction()
dy_numeric = NumericDiff(y)
d2y_numeric = NumericDiff(dy_numeric)

f = NumericDiff(dy_numeric)

start = -1.0
stop = 2.0

nodes = np.linspace(start, stop, num=51, endpoint=True)  # generate nodes
bc1 = y.evaluate(np.array([start]))[0]  # left Dirichlet boundary condition
bc2 = y.evaluate(np.array([stop]))[0]  # right Dirichlet boundary condition

result = dirichlet_poisson_solver(nodes, f, bc1, bc2, j=1.0)  # solve Poisson equation

dy_solution = np.gradient(result[:, 0], nodes, edge_order=2)
d2y_solution = np.gradient(dy_solution, nodes, edge_order=2)

# Plot the result
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
ax1.plot(nodes, f.evaluate(nodes), 'r-', label='f(x)')
ax1.plot(nodes, d2y_solution, 'b-', label='d2y/dx2 (solution)')
ax1.legend()

ax2.plot(nodes, dy_numeric.evaluate(nodes), 'r-', label='dy/dx')
ax2.plot(nodes, dy_solution, 'b-', label='dy/dx (solution)')
ax2.legend()

ax3.plot(nodes, result[:, 0], 'b-', label='solution')
ax3.plot(nodes, y.evaluate(nodes), 'r-', label='y(x)')
ax3.legend()

ax4.plot(nodes, result[:, 1], 'g-o', label='residual')
ax4.legend()
plt.show()
