import math as m
import numpy as np
from matplotlib import pyplot as plt

from BDPoisson1D import neumann_poisson_solver
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

start = 0.0
stop = 2.0
nodes = np.linspace(start, stop, num=5001, endpoint=True)
bc1 = dy_numeric.evaluate_point(start)
bc2 = dy_numeric.evaluate_point(stop)
integral = np.trapz(f.evaluate(nodes), nodes)
print(integral, bc2 - bc1)
print(np.allclose(integral, bc2 - bc1))

solution = neumann_poisson_solver(nodes, f, bc1, bc2, y0=y.evaluate_point(start))
dy_solution = NumericGradient(solution)
d2y_solution = NumericGradient(dy_solution)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
ax1.plot(nodes, np.asarray(f.evaluate(nodes)), 'r-', label='f(x)')
ax1.plot(nodes, np.asarray(d2y_solution.evaluate(nodes)), 'b-', label='d2y/dx2 (solution)')
ax1.legend()

ax2.plot(nodes, np.asarray(dy_numeric.evaluate(nodes)), 'r-', label='dy/dx')
ax2.plot(nodes, np.asarray(dy_solution.evaluate(nodes)), 'b-', label='dy/dx (solution)')
ax2.legend()

ax3.plot(nodes, np.asarray(solution.evaluate(nodes)), 'b-', label='solution')
ax3.plot(nodes, np.asarray(y.evaluate(nodes)), 'r-', label='y(x)')
ax3.legend()

ax4.plot(nodes, np.asarray(solution.error(nodes)), 'g-o', label='residual')
ax4.legend()
plt.show()
