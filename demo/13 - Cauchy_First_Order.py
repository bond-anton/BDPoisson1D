import math as m
import numpy as np
from matplotlib import pyplot as plt

from BDPoisson1D.FirstOrderLinear import cauchy_first_order_solver_arrays
from BDFunction1D import Function
from BDFunction1D.Differentiation import NumericGradient


class TestFunction(Function):
    """
    Some known differentiable function
    """
    def evaluate_point(self, x):
        # return -10 * m.sin(2 * m.pi * x**2) / (2 * m.pi) + 3 * x**2 + x + 5
        return m.sin(x) + 5


class TestDerivative(Function):
    """
    Some known differentiable function
    """
    def evaluate_point(self, x):
        # return -20 * x * m.cos(2 * m.pi * x**2) + 6 * x + 1
        return m.cos(x)


y = TestFunction()
dy_numeric = NumericGradient(y)
dy_analytic = TestDerivative()

dy = dy_analytic

start = -1.0
stop = 2.0

hundreds = []
errs = []

fig, ax = plt.subplots()

for i in range(1, 20):
    nodes = np.linspace(start, stop, num=100 * i + 1, endpoint=True)  # generate nodes
    idx = 0
    bc = y.evaluate_point(nodes[idx])  # left Dirichlet boundary condition
    f_nodes = np.asarray(dy.evaluate(nodes))
    result = cauchy_first_order_solver_arrays(nodes, f_nodes, bc, idx, j=1.0)  # solve Poisson equation
    ax.plot(nodes, np.asarray(result))
    dy_solution = np.gradient(result[:], nodes, edge_order=2)
    hundreds.append(i)
    errs.append(np.sqrt(np.square(result[2:-2] - np.asarray(y.evaluate(nodes[2:-2]))).mean()))
    print(101 + i * 100, 'Mean Square ERR:', errs[-1])

plt.show()

fig, ax = plt.subplots()
ax.plot(hundreds, errs, 'r-o')
ax.set_xlabel('Points number x100')
ax.set_ylabel('Mean Square Error')
plt.show()

# Plot the result
fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
ax1.plot(nodes, f_nodes, 'r-', label='f(x)')
ax1.plot(nodes, dy_solution, 'b-', label='dy/dx (solution)')
ax1.legend()

ax2.plot(nodes, np.asarray(y.evaluate(nodes)), 'r-', label='y(x)')
ax2.plot(nodes, np.asarray(result[:]), 'b-', label='solution')
ax2.legend()

ax3.plot(nodes[2:-2], result[2:-2] - np.asarray(y.evaluate(nodes[2:-2])), 'g-o', label='residual')
ax3.legend()
plt.show()
