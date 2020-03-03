import numpy as np
from matplotlib import pyplot as plt

from BDPoisson1D.FirstOrderLinear import dirichlet_first_order_solver_amr
from BDPoisson1D import Function, NumericGradient
from BDMesh import Mesh1DUniform


class TestFunction(Function):
    """
    Some known differentiable function
    """
    def evaluate(self, x):
        xx = np.asarray(x)
        return -10 * np.sin(2 * np.pi * xx**2) / (2 * np.pi) + 3 * xx ** 2 + xx + 5


class TestDerivative(Function):
    """
    Some known differentiable function
    """
    def evaluate(self, x):
        xx = np.asarray(x)
        return -20 * xx * np.cos(2 * np.pi * xx**2) + 6 * xx + 1


class MixFunction(Function):
    """
    Some known differentiable function
    """
    def evaluate(self, x):
        xx = np.asarray(x)
        return xx**2 * np.cos(xx)


class FFunction(Function):
    """
    Some known differentiable function
    """
    def __init__(self):
        super(FFunction, self).__init__()
        self.y = TestFunction()
        self.dy = TestDerivative()
        self.p = MixFunction()

    def evaluate(self, x):
        return self.y.evaluate(x) * self.p.evaluate(x) + self.dy.evaluate(x)


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


y = TestFunction()
p = MixFunction()
f = FFunction()
dy_numeric = NumericGradient(y)
dy_analytic = TestDerivative()

dy = dy_analytic

start = -1.0
stop = 2.0
bc1 = y.evaluate(np.array([start]))[0]  # left Dirichlet boundary condition
bc2 = y.evaluate(np.array([stop]))[0]  # right Dirichlet boundary condition


step = (stop - start) / (100 * 1)
meshes = dirichlet_first_order_solver_amr(start, stop, step, p, f, bc1, bc2,
                                          max_iter=1000, threshold=1e-2, max_level=20)
flat_mesh = meshes.flatten()
dy_solution = np.gradient(flat_mesh.solution, flat_mesh.physical_nodes, edge_order=2)

# Plot the result
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True)
ax1.plot(flat_mesh.physical_nodes, f.evaluate(flat_mesh.physical_nodes), 'r-', label='f(x)')
ax1.plot(flat_mesh.physical_nodes,
         dy_solution + np.asarray(flat_mesh.solution) * np.asarray(p.evaluate(flat_mesh.physical_nodes)),
         'b-', label='dy/dx + p(x)*y (solution)')
ax1.legend()

ax2.plot(flat_mesh.physical_nodes, dy_numeric.evaluate(flat_mesh.physical_nodes), 'g-', label='dy/dx')
ax2.plot(flat_mesh.physical_nodes, dy_analytic.evaluate(flat_mesh.physical_nodes), 'r-', label='dy/dx')
ax2.plot(flat_mesh.physical_nodes, dy_solution, 'b-', label='dy/dx (solution)')
ax2.legend()

ax3.plot(flat_mesh.physical_nodes, y.evaluate(flat_mesh.physical_nodes), 'r-', label='y(x)')
ax3.plot(flat_mesh.physical_nodes, flat_mesh.solution, 'b-', label='solution')
ax3.legend()

plot_tree(meshes, ax4)

ax5.plot(flat_mesh.physical_nodes, flat_mesh.residual, 'g-o', label='residual')
ax5.legend()

plt.show()
