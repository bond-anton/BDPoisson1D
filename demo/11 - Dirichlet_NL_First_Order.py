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

w = 1.0
min_w = 0.3
mse_threshold = 1e-80
i = 0
max_iterations = 100
mse_old = 1e20
while i < max_iterations:
    # result = dirichlet_non_linear_first_order_solver_arrays(nodes, y_nodes, p_nodes,
    #                                                         f_nodes, df_dy_nodes,
    #                                                         bc1, bc2, j=1.0, w=w)
    result = dirichlet_non_linear_first_order_solver(nodes, y, p, f, df_dy, bc1, bc2, j=1.0, w=w)

    y = InterpolateFunction(nodes, result[:, 0])

    f.f = y
    df_dy.f = y

    y_nodes = y.evaluate(nodes)
    p_nodes = p.evaluate(nodes)
    f_nodes = f.evaluate(nodes)
    df_dy_nodes = df_dy.evaluate(nodes)

    mse = np.square(result[:, 1]).mean()
    if mse > mse_old:
        if w > min_w:
            w -= 0.1
            print(i, ' -> reduced W to', w)
            continue
        else:
            print('Not converging anymore. W =', w)
            break
    if mse < mse_threshold:
        break

    mse_old = mse
    i += 1
print('Reached MSE:', mse, 'in', i, 'iterations.')

dy_solution = np.gradient(result[:, 0], nodes, edge_order=2)
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

ax4.plot(nodes, result[:, 1], 'g-o', label='residual')
ax4.legend()
plt.show()

print('Compare to analytic solution')
print('MSE:', np.square(np.asarray(result[:, 0]) - np.asarray(y.evaluate(nodes[:]))).mean())

fig, ax = plt.subplots()
ax.plot(nodes, np.asarray(result[:, 0]) - np.asarray(y.evaluate(nodes[:])), 'g-o', label='residual')
plt.show()
