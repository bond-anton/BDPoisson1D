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
        return np.sin(xx) ** 2


class TestFunctional(Functional):
    """
    f(x, y), RHS of the ODE
    """

    def evaluate(self, x):
        xx = np.asarray(x)
        yy = np.asarray(self.f.evaluate(x))
        result = np.empty_like(yy)
        idc = np.where(yy >= 0.5)
        ids = np.where(yy < 0.5)
        result[ids] = 2 * np.sign(np.cos(xx[ids])) * np.sin(xx[ids]) * np.sqrt(1 - yy[ids])
        result[idc] = 2 * np.sign(np.sin(xx[idc])) * np.cos(xx[idc]) * np.sqrt(yy[idc])
        return result


class TestFunctionalDf(Functional):
    """
    df/dy(x, y)
    """

    def evaluate(self, x):
        xx = np.asarray(x)
        yy = np.asarray(self.f.evaluate(x))
        result = np.empty_like(yy)
        idc = np.where(yy >= 0.5)
        ids = np.where(yy < 0.5)
        result[ids] = -np.sign(np.cos(xx[ids])) * np.sin(xx[ids]) / np.sqrt(1 - yy[ids])
        result[idc] = np.sign(np.sin(xx[idc])) * np.cos(xx[idc]) / np.sqrt(yy[idc])
        return result


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


y0 = TestFunction()
dy0_numeric = NumericGradient(y0)
shift = np.pi * 11 + 1
start = -3 * np.pi / 2 + shift
stop = 3 * np.pi / 2 + shift + 0.5
nodes = np.linspace(start, stop, num=1001, endpoint=True)  # generate nodes
bc1 = y0.evaluate([nodes[0]])[0]
bc2 = y0.evaluate([nodes[-1]])[0]

y = GuessFunction()
p = MixFunction()

f = TestFunctional(y)
df_dy = TestFunctionalDf(y)

y_nodes = y.evaluate(nodes)
p_nodes = p.evaluate(nodes)
f_nodes = f.evaluate(nodes)
df_dy_nodes = df_dy.evaluate(nodes)

for i in range(100):
    # result = dirichlet_non_linear_first_order_solver_arrays(nodes, y_nodes, p_nodes,
    #                                                         f_nodes, df_dy_nodes,
    #                                                         bc1, bc2, j=1.0, w=0.7)
    result = dirichlet_non_linear_first_order_solver(nodes, y, p, f, df_dy, bc1, bc2, j=1.0, w=0.7)
    y = InterpolateFunction(nodes, result[:, 0])
    dy_solution = np.gradient(result[:, 0], nodes, edge_order=2)
    f.f = y
    df_dy.f = y

    y_nodes = y.evaluate(nodes)
    p_nodes = p.evaluate(nodes)
    f_nodes = f.evaluate(nodes)
    df_dy_nodes = df_dy.evaluate(nodes)

# Plot the result
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
ax1.plot(nodes, f.evaluate(nodes), 'r-', label='f(x)')
ax1.plot(nodes, dy_solution + np.asarray(result[:, 0]) * np.asarray(p.evaluate(nodes)),
         'b-', label='dy/dx + p(x)*y (solution)')
ax1.legend()

ax2.plot(nodes, dy0_numeric.evaluate(nodes), 'r-', label='dy/dx')
ax2.plot(nodes, dy_solution, 'b-', label='dy/dx (solution)')
ax2.legend()

ax3.plot(nodes, y0.evaluate(nodes), 'r-', label='y(x)')
ax3.plot(nodes, result[:, 0], 'b-', label='solution')
ax3.legend()

ax4.plot(nodes, result[:, 2], 'g-o', label='residual')
ax4.legend()
plt.show()
