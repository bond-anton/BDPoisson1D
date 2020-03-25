import math as m
import numpy as np
from matplotlib import pyplot as plt

from BDPoisson1D.FirstOrderNonLinear import dirichlet_non_linear_first_order_solver_arrays
from BDPoisson1D.FirstOrderNonLinear import dirichlet_non_linear_first_order_solver
from BDFunction1D import Function
from BDFunction1D.Functional import Functional
from BDFunction1D.Differentiation import NumericGradient
from BDFunction1D.Interpolation import InterpolateFunction


class TestFunction(Function):
    """
    Some known differentiable function
    """

    def evaluate_point(self, x):
        return m.sin(x) ** 2


class TestFunctional(Functional):
    """
    f(x, y), RHS of the ODE
    """

    def evaluate_point(self, x):
        y = self.f.evaluate_point(x)
        if y >= 0.5:
            return 2 * np.sign(m.sin(x)) * m.cos(x) * m.sqrt(y)
        else:
            return 2 * np.sign(m.cos(x)) * m.sin(x) * m.sqrt(1 - y)


class TestFunctionalDf(Functional):
    """
    df/dy(x, y)
    """

    def evaluate_point(self, x):
        y = self.f.evaluate_point(x)
        if y >= 0.5:
            return np.sign(m.sin(x)) * m.cos(x) / m.sqrt(y)
        else:
            return -np.sign(m.cos(x)) * m.sin(x) / m.sqrt(1 - y)


class MixFunction(Function):
    """
    Some known differentiable function
    """

    def evaluate_point(self, x):
        return 0.0


class GuessFunction(Function):
    """
    Some known differentiable function
    """

    def evaluate_point(self, x):
        return 0.0


y0 = TestFunction()
dy0_numeric = NumericGradient(y0)
shift = np.pi * 11 + 1
start = -3 * np.pi / 2 + shift
stop = 3 * np.pi / 2 + shift + 0.5
bc1 = y0.evaluate_point(start)
bc2 = y0.evaluate_point(stop)
y = GuessFunction()
p = MixFunction()
f = TestFunctional(y)
df_dy = TestFunctionalDf(y)
w = 0.7
hundreds = []
errs = []
for i in range(1, 200):
    nodes = np.linspace(start, stop, num=100 * i + 1, endpoint=True)  # generate nodes
    y_nodes = y.evaluate(nodes)
    p_nodes = p.evaluate(nodes)
    f_nodes = f.evaluate(nodes)
    df_dy_nodes = df_dy.evaluate(nodes)

    # result = dirichlet_non_linear_first_order_solver_arrays(nodes, y_nodes, p_nodes,
    #                                                         f_nodes, df_dy_nodes,
    #                                                         bc1, bc2, j=1.0, w=w)
    result = dirichlet_non_linear_first_order_solver(nodes, y, p, f, df_dy, bc1, bc2, j=1.0, w=w)
    mse = np.sqrt(np.square(result[:, 1]).mean())
    hundreds.append(i)
    errs.append(mse)
    print(101 + i * 100, 'Mean Square ERR:', errs[-1])
errs = np.array(errs)
mean_errs = errs.mean()
hundreds = np.array(hundreds)
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(hundreds, errs, 'r-o')
ax1.plot(hundreds[[0, -1]], [mean_errs] * 2, 'b-')
ax1.set_xlabel('Points number x100')
ax1.set_ylabel('Mean Square Error')

bin_size = 5
bins = []
std_errs = []
i = 0
while i < hundreds.shape[0] - 1:
    if i + bin_size < hundreds.shape[0] - 1:
        di = bin_size
    else:
        di = hundreds.shape[0] - 1 - i
    bins.append(hundreds[i:i + di].mean())
    std_errs.append(np.sqrt(np.square(errs[i:i + di] - mean_errs).mean()))
    i += di
ax2.plot(bins, std_errs, 'r-o')
ax2.set_xlabel('Points number x100')
ax2.set_ylabel('Mean Square Error STD')
plt.show()

nodes = np.linspace(start, stop, num=1001, endpoint=True)  # generate nodes
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

    mse = np.sqrt(np.square(result[:, 1]).mean())
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
