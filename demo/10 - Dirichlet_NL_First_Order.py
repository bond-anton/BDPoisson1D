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
        return -10 * np.sin(2 * np.pi * np.asarray(x)**2) / (2 * np.pi) + 3 * np.asarray(x) ** 2 + np.asarray(x) + 5
        # return np.ones(x.shape[0], dtype=np.double)
        # return np.ones(x.shape[0], dtype=np.double)

class TestFunctional(Functional):
    def __init__(self, Nd, kT, f):
        super(TestFunctional, self).__init__(f)
        self.Nd = Nd
        self.kT = kT

    def evaluate(self, x):
        return self.Nd(np.asarray(x)) * (1 - (np.exp(-np.asarray(self.f.evaluate(x)) / self.kT)))
        # return self.Nd(np.asarray(x)) / self.kT * np.asarray(self.f.evaluate(x))**2


class TestFunctionalDf(Functional):
    def __init__(self, Nd, kT, f):
        super(TestFunctionalDf, self).__init__(f)
        self.Nd = Nd
        self.kT = kT

    def evaluate(self, x):
        return self.Nd(np.asarray(x)) / self.kT * np.exp(-np.asarray(self.f.evaluate(x)) / self.kT)
        # return 2 * self.Nd(np.asarray(x)) / self.kT * np.asarray(self.f.evaluate(x))


class MixFunction(Function):
    """
    Some known differentiable function
    """
    def evaluate(self, x):
        # return np.ones(x.shape[0], dtype=np.double)
        # return np.zeros(x.shape[0], dtype=np.double)
        return np.asarray(x)**2 * np.cos(np.asarray(x))


Nd = lambda x: np.ones_like(x)
kT = 1.0e-2

y = TestFunction()
dy_numeric = NumericGradient(y)
p = MixFunction()

f = TestFunctional(Nd, kT, y)
df_dy = TestFunctionalDf(Nd, kT, y)

start = -1
stop = 2

nodes = np.linspace(start, stop, num=103, endpoint=True)  # generate nodes
y0_nodes = y.evaluate(nodes)
p_nodes = p.evaluate(nodes)
f_nodes = f.evaluate(nodes)
df_dy_nodes = df_dy.evaluate(nodes)

bc1 = 1.0
bc2 = 3.7

for i in range(5000):
    # result = dirichlet_non_linear_first_order_solver_arrays(nodes, y0_nodes, p_nodes,
    #                                                         f_nodes, df_dy_nodes,
    #                                                         bc1, bc2, j=1.0, w=1.0)
    result = dirichlet_non_linear_first_order_solver(nodes, y, p, f, df_dy, bc1, bc2, j=1.0, w=1.0)
    y = InterpolateFunction(nodes, result[:, 0])
    f.f = y
    df_dy.f = y
    y0_nodes = y.evaluate(nodes)
    p_nodes = p.evaluate(nodes)
    f_nodes = f.evaluate(nodes)
    df_dy_nodes = df_dy.evaluate(nodes)


dy_solution = np.gradient(result[:, 0], nodes, edge_order=2)

# Plot the result
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
ax1.plot(nodes, f_nodes, 'r-', label='f(x)')
ax1.plot(nodes[2:-2], dy_solution[2:-2] + np.asarray(result[2:-2, 0]) * np.asarray(p_nodes[2:-2]), 'b-', label='dy/dx + p(x)*y (solution)')
ax1.legend()

ax2.plot(nodes, dy_numeric.evaluate(nodes), 'r-', label='dy/dx')
ax2.plot(nodes[:], dy_solution[:], 'b-', label='dy/dx (solution)')
ax2.legend()

ax3.plot(nodes, y.evaluate(nodes), 'r-', label='y(x)')
ax3.plot(nodes[:], result[:, 0], 'b-', label='solution')
ax3.legend()

ax4.plot(nodes[:], result[:, 1], 'g-o', label='residual')
ax4.legend()
plt.show()
print(np.mean(result[:, 1]))
