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
            axes.plot(mesh.physical_nodes, np.ones(mesh.num) * level, colors[level] + styles[i] + 'x')
    axes.set_ylim([-1, max(mesh_tree.tree.keys()) + 1])
    axes.grid()


start = 0.2
stop = 1.2
step = 0.02
meshes = dirichlet_poisson_solver_amr(start, stop, step, f,
                                      y.evaluate_point(start),
                                      y.evaluate_point(stop),
                                      max_iter=100,
                                      threshold=1.0e-4, max_level=15)
flat_mesh = meshes.flatten()
dy_solution = np.gradient(flat_mesh.solution, flat_mesh.physical_nodes, edge_order=1)
d2y_solution = np.gradient(dy_solution, flat_mesh.physical_nodes, edge_order=1)

# Plot the result
fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True)
ax[0][0].plot(flat_mesh.physical_nodes, d2y_solution, 'b-', label='d2y/dx2 (solution)')
ax[0][0].plot(flat_mesh.physical_nodes, f.evaluate(flat_mesh.physical_nodes), 'r-', label='f(x)')
ax[0][1].plot(flat_mesh.physical_nodes, flat_mesh.solution, 'b-', label='solution')
ax[0][1].plot(flat_mesh.physical_nodes, y.evaluate(flat_mesh.physical_nodes), 'r-', label='y(x)')
ax[1][0].plot(flat_mesh.physical_nodes, flat_mesh.residual, 'g-o', label='residual')
plot_tree(meshes, ax[1][1])
ax[0][0].legend()
ax[0][1].legend()
ax[1][0].legend()
plt.show()
