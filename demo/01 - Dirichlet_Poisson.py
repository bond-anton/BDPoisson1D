import math as m
import numpy as np
from matplotlib import pyplot as plt

from BDPoisson1D import dirichlet_poisson_solver
from BDFunction1D import Function
from BDFunction1D.Differentiation import NumericGradient


class TestFunction(Function):
    """
    Some known differentiable function
    """
    def evaluate_point(self, x):
        return -10 * m.sin(m.pi * x**2) / (2 * m.pi) + 3 * x**2 + x + 5

y = TestFunction()
dy_numeric = NumericGradient(y)
d2y_numeric = NumericGradient(dy_numeric)

f = NumericGradient(dy_numeric)

start = -1.0
stop = 2.0

nodes = np.linspace(start, stop, num=51, endpoint=True)  # generate nodes
bc1 = y.evaluate_point(start)  # left Dirichlet boundary condition
bc2 = y.evaluate_point(stop)  # right Dirichlet boundary condition

solution = dirichlet_poisson_solver(nodes, f, bc1, bc2, j=1.0)  # solve Poisson equation

dy_solution = NumericGradient(solution)
d2y_solution = NumericGradient(dy_solution)

# Plot the result
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
ax1.plot(nodes, f.evaluate(nodes), 'r-', label='f(x)')
ax1.plot(nodes, d2y_solution.evaluate(nodes), 'b-', label='d2y/dx2 (solution)')
ax1.legend()

ax2.plot(nodes, dy_numeric.evaluate(nodes), 'r-', label='dy/dx')
ax2.plot(nodes, dy_solution.evaluate(nodes), 'b-', label='dy/dx (solution)')
ax2.legend()

ax3.plot(nodes, solution.evaluate(nodes), 'b-', label='solution')
ax3.plot(nodes, y.evaluate(nodes), 'r-', label='y(x)')
ax3.legend()

ax4.plot(nodes, solution.error(nodes), 'g-o', label='residual')
ax4.legend()
plt.show()
