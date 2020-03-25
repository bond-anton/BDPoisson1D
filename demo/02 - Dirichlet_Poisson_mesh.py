import math as m
import numpy as np
from matplotlib import pyplot as plt

from BDPoisson1D import dirichlet_poisson_solver_mesh
from BDMesh import Mesh1DUniform
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
mesh = Mesh1DUniform(start, stop,
                     boundary_condition_1=y.evaluate_point(start),
                     boundary_condition_2=y.evaluate_point(stop),
                     physical_step=0.02)

dirichlet_poisson_solver_mesh(mesh, f)  # solve Poisson equation

dy_solution = np.gradient(mesh.solution, mesh.physical_nodes, edge_order=2)
d2y_solution = np.gradient(dy_solution, mesh.physical_nodes, edge_order=2)

# Plot the result
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
ax1.plot(mesh.physical_nodes, f.evaluate(mesh.physical_nodes), 'r-', label='f(x)')
ax1.plot(mesh.physical_nodes, d2y_solution, 'b-', label='d2y/dx2 (solution)')
ax1.legend()

ax2.plot(mesh.physical_nodes, dy_numeric.evaluate(mesh.physical_nodes), 'r-', label='dy/dx')
ax2.plot(mesh.physical_nodes, dy_solution, 'b-', label='dy/dx (solution)')
ax2.legend()

ax3.plot(mesh.physical_nodes, mesh.solution, 'b-', label='solution')
ax3.plot(mesh.physical_nodes, y.evaluate(mesh.physical_nodes), 'r-', label='y(x)')
ax3.legend()

ax4.plot(mesh.physical_nodes, mesh.residual, 'g-o', label='residual')
ax4.legend()
plt.show()
