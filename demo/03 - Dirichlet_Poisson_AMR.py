import math as m
import numpy as np
from matplotlib import pyplot as plt

from BDPoisson1D import dirichlet_poisson_solver_amr
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


def plot_tree(mesh_tree, axes=None):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b', 'c', 'm', 'y', 'k',
              'r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b', 'c', 'm', 'y', 'k']
    styles = ['-', ':', '--', '-', ':', '--', '-', '-', ':', '--', '-', ':', '--', '-',
              '-', ':', '--', '-', ':', '--', '-', '-', ':', '--', '-', ':', '--', '-',
              '-', ':', '--', '-', ':', '--', '-', '-', ':', '--', '-', ':', '--', '-',
              '-', ':', '--', '-', ':', '--', '-', '-', ':', '--', '-', ':', '--', '-']
    if axes is None:
        _, axes = plt.subplots()
    for level in mesh_tree.levels:
        for i, mesh in enumerate(mesh_tree.tree[level]):
            axes.plot(np.asarray(mesh.physical_nodes), np.ones(mesh.num) * level, colors[level] + styles[i] + 'x')
    axes.set_ylim([-1, max(mesh_tree.tree.keys()) + 1])
    axes.grid()


start = 0.2
stop = 1.2
step = 0.02
solution = dirichlet_poisson_solver_amr(start, stop, step, f,
                                        y.evaluate_point(start),
                                        y.evaluate_point(stop),
                                        max_iter=100,
                                        threshold=1.0e-4, max_level=15)
dy_solution = NumericGradient(solution)
d2y_solution = NumericGradient(dy_solution)

# Plot the result
fig, ax = plt.subplots(nrows=3, sharex=True)

nodes = np.linspace(start, stop, num=int((stop-start)/step)+1)

ax[0].plot(nodes, np.asarray(d2y_solution.evaluate(nodes)), 'b-', label='d2y/dx2 (solution)')
ax[0].plot(nodes, np.asarray(f.evaluate(nodes)), 'r-', label='f(x)')
ax[1].plot(nodes, np.asarray(solution.evaluate(nodes)), 'b-', label='solution')
ax[1].plot(nodes, np.asarray(y.evaluate(nodes)), 'r-', label='y(x)')
ax[2].plot(nodes, np.asarray(solution.error(nodes)), 'g-o', label='residual')
ax[0].legend()
ax[1].legend()
ax[2].legend()
plt.show()
