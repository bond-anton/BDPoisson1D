import math as m
import numpy as np
from matplotlib import pyplot as plt

from BDPoisson1D.FirstOrderLinear import dirichlet_first_order_solver_mesh
from BDMesh import Mesh1DUniform
from BDFunction1D import Function
from BDFunction1D.Differentiation import NumericGradient


class TestFunction(Function):
    """
    Some known differentiable function
    """
    def evaluate_point(self, x):
        return -10 * m.sin(2 * m.pi * x**2) / (2 * m.pi) + 3 * x**2 + x + 5


class TestDerivative(Function):
    """
    Some known differentiable function
    """
    def evaluate_point(self, x):
        return -20 * x * m.cos(2 * m.pi * x**2) + 6 * x + 1


class MixFunction(Function):
    """
    Some known differentiable function
    """
    def evaluate_point(self, x):
        # return 1.0
        # return 0.0
        return x**2 * m.cos(x)


class FFunction(Function):
    """
    Some known differentiable function
    """
    def __init__(self):
        super(FFunction, self).__init__()
        self.y = TestFunction()
        self.dy = TestDerivative()
        self.p = MixFunction()

    def evaluate_point(self, x):
        return self.y.evaluate_point(x) * self.p.evaluate_point(x) + self.dy.evaluate_point(x)

y = TestFunction()
p = MixFunction()
f = FFunction()
dy_numeric = NumericGradient(y)
dy_analytic = TestDerivative()

dy = dy_analytic

start = -1.0
stop = 2.0
bc1 = y.evaluate_point(start)  # left Dirichlet boundary condition
bc2 = y.evaluate_point(stop)  # right Dirichlet boundary condition

hundreds = []
errs = []
for i in range(2, 20):
    root_mesh = Mesh1DUniform(start, stop, bc1, bc2, num=100*i + 1)
    dirichlet_first_order_solver_mesh(root_mesh, p, f)
    dy_solution = np.gradient(root_mesh.solution, root_mesh.physical_nodes, edge_order=2)
    hundreds.append(i)
    errs.append(np.square(np.asarray(y.evaluate(root_mesh.physical_nodes)) - root_mesh.solution).mean())
    print(101 + i * 100, 'Mean Square ERR:', errs[-1])

fig, ax = plt.subplots()
ax.plot(hundreds, errs, 'r-o')
ax.set_xlabel('Points number x100')
ax.set_ylabel('Mean Square Error')
plt.show()

# Plot the result
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
ax1.plot(np.asarray(root_mesh.physical_nodes), np.asarray(f.evaluate(root_mesh.physical_nodes)), 'r-', label='f(x)')
ax1.plot(np.asarray(root_mesh.physical_nodes),
         np.asarray(dy_solution) + np.asarray(root_mesh.solution) * np.asarray(p.evaluate(root_mesh.physical_nodes)),
         'b-', label='dy/dx + p(x)*y (solution)')
ax1.legend()

ax2.plot(np.asarray(root_mesh.physical_nodes), np.asarray(dy_numeric.evaluate(root_mesh.physical_nodes)),
         'g-', label='dy/dx')
ax2.plot(np.asarray(root_mesh.physical_nodes), np.asarray(dy_analytic.evaluate(root_mesh.physical_nodes)),
         'r-', label='dy/dx')
ax2.plot(np.asarray(root_mesh.physical_nodes), np.asarray(dy_solution), 'b-', label='dy/dx (solution)')
ax2.legend()

ax3.plot(np.asarray(root_mesh.physical_nodes), np.asarray(y.evaluate(root_mesh.physical_nodes)), 'r-', label='y(x)')
ax3.plot(np.asarray(root_mesh.physical_nodes), np.asarray(root_mesh.solution), 'b-', label='solution')
ax3.legend()

ax4.plot(np.asarray(root_mesh.physical_nodes), np.asarray(y.evaluate(root_mesh.physical_nodes)) - root_mesh.solution,
         'g-o', label='residual')
ax4.legend()
plt.show()
