import math as m
import numpy as np
from matplotlib import pyplot as plt

from BDPoisson1D import neumann_poisson_solver, neumann_poisson_solver_amr
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
step = 0.002
threshold = 1.0e-2
max_level = 10
nodes = np.linspace(start, stop, num=1001, endpoint=True)
print('STEP:', nodes[1] - nodes[0])
bc1 = dy_numeric.evaluate_point(start)
bc2 = dy_numeric.evaluate_point(stop)
integral = np.trapz(f.evaluate(nodes), nodes)
print(integral, bc2 - bc1)
print(np.allclose(integral, bc2 - bc1))
print('Y0:', y.evaluate(np.asarray([start]))[0])

meshes = neumann_poisson_solver_amr(start, stop, step, f, bc1, bc2,
                                    y.evaluate_point(start),
                                    25,
                                    threshold, max_level=max_level)

flat_mesh = meshes.flatten()
result = np.vstack((flat_mesh.solution, flat_mesh.residual)).T
nodes = flat_mesh.physical_nodes

# result = neumann_poisson_solver(nodes, f, bc1, bc2, y0=y.evaluate(np.asarray([start]))[0])
dy_solution = np.gradient(result[:, 0], nodes, edge_order=1)
d2y_solution = np.gradient(dy_solution, nodes, edge_order=1)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
ax1.plot(nodes, f.evaluate(nodes), 'r-', label='f(x)')
ax1.plot(nodes, d2y_solution, 'b-', label='d2y/dx2 (solution)')
ax1.legend()

ax2.plot(nodes, dy_numeric.evaluate(nodes), 'r-', label='dy/dx')
ax2.plot(nodes, dy_solution, 'b-', label='dy/dx (solution)')
ax2.legend()

ax3.plot(nodes[260:], result[260:, 0], 'b-', label='solution')
ax3.plot(nodes[:], y.evaluate(nodes)[:], 'r-', label='y(x)')
ax3.legend()

ax4.plot(nodes, result[:, 1], 'g-o', label='residual')
ax4.legend()
plt.show()
