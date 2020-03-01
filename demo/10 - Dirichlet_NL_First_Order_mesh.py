import numpy as np
from matplotlib import pyplot as plt

from BDPoisson1D.FirstOrderNonLinear import dirichlet_non_linear_first_order_solver_arrays
from BDPoisson1D.FirstOrderNonLinear import dirichlet_non_linear_first_order_solver
from BDPoisson1D import Function, Functional, NumericGradient, InterpolateFunction


class TestFunction(Function):
    """
    Some known differentiable function
    """
    def evaluate(self, x):
        xx = np.asarray(x)
        return np.sin(xx)**2


class TestFunctional(Functional):
    """
    f(x, y), RHS of the ODE
    """
    def evaluate(self, x):
        xx = np.asarray(x)
        yy = np.asarray(self.f.evaluate(x))
        return 2 * np.sign(np.cos(xx)) * np.sin(xx) * np.sqrt(1 - yy)


class TestFunctionalDf(Functional):
    """
    df/dy(x, y)
    """
    def evaluate(self, x):
        xx = np.asarray(x)
        yy = np.asarray(self.f.evaluate(x))
        return -np.sign(np.cos(xx)) * np.sin(xx) / np.sqrt(1 - yy)


class MixFunction(Function):
    """
    Some known differentiable function
    """
    def evaluate(self, x):
        return np.zeros(x.shape[0], dtype=np.double)


class GuessFunction(Function):
    """
    Some known differentiable function
    """
    def evaluate(self, x):
        return np.zeros(x.shape[0], dtype=np.double)
        # return 0.9*np.ones(x.shape[0], dtype=np.double)
        # return np.sin(np.asarray(x))**2


y0 = TestFunction()
dy0_numeric = NumericGradient(y0)
eps = 1e-1
shift = np.pi * 11
start = -3*np.pi/2 + shift + eps
stop = 3*np.pi/2 + shift - eps
nodes = np.linspace(start, stop, num=1001, endpoint=True)  # generate nodes
bc1 = y0.evaluate([nodes[0]])[0]
bc2 = y0.evaluate([nodes[-1]])[0]
print(bc1, bc2)

y = GuessFunction()
p = MixFunction()

f = TestFunctional(y)
df_dy = TestFunctionalDf(y)

for i in range(100):
    result = dirichlet_non_linear_first_order_solver(nodes, y, p, f, df_dy, bc1, bc2, j=1.0, w=0.5)
    yy = np.asarray(result)
    yy[np.where(yy[:, 0] >= 1), 0] = 1 - 1e-1
    # y = InterpolateFunction(nodes, result[:, 0])
    y = InterpolateFunction(nodes, yy[:, 0])
    # dy_solution = np.gradient(result[:, 0], nodes, edge_order=2)
    dy_solution = np.gradient(yy[:, 0], nodes, edge_order=2)
    f.f = y
    df_dy.f = y

c1 = 0
c2 = -1
# Plot the result
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
ax1.plot(nodes, f.evaluate(nodes), 'r-', label='f(x)')
ax1.plot(nodes[c1:c2], dy_solution[c1:c2] + np.asarray(result[c1:c2, 0]) * np.asarray(p.evaluate(nodes[c1:c2])),
         'b-', label='dy/dx + p(x)*y (solution)')
ax1.legend()

ax2.plot(nodes, dy0_numeric.evaluate(nodes), 'r-', label='dy/dx')
ax2.plot(nodes[c1:c2], dy_solution[c1:c2], 'b-', label='dy/dx (solution)')
ax2.legend()

ax3.plot(nodes, y0.evaluate(nodes), 'r-', label='y(x)')
ax3.plot(nodes[c1:c2], result[c1:c2, 0], 'b-', label='solution')
ax3.legend()

ax4.plot(nodes[c1:c2], result[c1:c2, 1], 'g-o', label='residual')
ax4.legend()
plt.show()
print(np.mean(result[:, 1]))
