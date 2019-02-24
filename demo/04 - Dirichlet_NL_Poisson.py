import numpy as np
from matplotlib import pyplot as plt

from BDPoisson1D import dirichlet_non_linear_poisson_solver
from BDPoisson1D import Function, Functional, InterpolateFunction


class TestFunction(Function):
    """
    Some known differentiable function
    """
    def evaluate(self, x):
        return np.exp(-np.asarray(x) * 3)


class TestFunctional(Functional):
    def __init__(self, Nd, kT, f):
        super(TestFunctional, self).__init__(f)
        self.Nd = Nd
        self.kT = kT

    def evaluate(self, x):
        return self.Nd(np.asarray(x)) * (1 - (np.exp(-np.asarray(self.f.evaluate(x)) / self.kT)))


class TestFunctionalDf(Functional):
    def __init__(self, Nd, kT, f):
        super(TestFunctionalDf, self).__init__(f)
        self.Nd = Nd
        self.kT = kT

    def evaluate(self, x):
        return self.Nd(np.asarray(x)) / self.kT * np.exp(-np.asarray(self.f.evaluate(x)) / self.kT)


Nd = lambda x: np.ones_like(x)
kT = 1 / 20

Psi = TestFunction()
f = TestFunctional(Nd, kT, Psi)
dfdDPsi = TestFunctionalDf(Nd, kT, Psi)

nodes = np.linspace(0., 4., num=21, endpoint=True, dtype=np.float)
bc1 = 1.0
bc2 = 0.0

print(nodes, nodes.size, nodes.dtype)

DPsi = np.zeros_like(nodes)
E = np.zeros_like(nodes)
_, (ax1, ax2, ax3, ax4) = plt.subplots(4)
ax1.set_autoscaley_on(True)
ax2.set_autoscaley_on(True)
ax3.set_autoscaley_on(True)
ax4.set_autoscaley_on(True)
Psi_line, = ax1.plot(nodes, Psi.evaluate(nodes))
DPsi_line, = ax2.plot(nodes, DPsi)
f_line, = ax3.plot(nodes, f.evaluate(nodes))
E_line, = ax4.plot(nodes, E)
print(Psi.evaluate(nodes), Psi.evaluate(nodes).size, Psi.evaluate(nodes).dtype)

dPsi = np.gradient(Psi.evaluate(nodes), nodes, edge_order=2)
print(dPsi, dPsi.size)
d2Psi = np.gradient(dPsi, nodes, edge_order=2)
print(d2Psi, d2Psi.size)
d2Psi_line, = ax3.plot(nodes, d2Psi)


plt.draw()


for i in range(100):
    print(i + 1)
    result = dirichlet_non_linear_poisson_solver(nodes, Psi, f, dfdDPsi, bc1=bc1, bc2=bc2, j=1.0)
    Psi = InterpolateFunction(nodes, result[:, 0])
    f.f = Psi
    dfdDPsi.f = Psi
    DPsi = result[:, 1]
    R = result[:, 2]
    dPsi = np.gradient(Psi.evaluate(nodes), nodes, edge_order=2)
    d2Psi = np.gradient(dPsi, nodes, edge_order=2)
    Psi_line.set_ydata(Psi.evaluate(nodes))
    DPsi_line.set_ydata(DPsi)
    f_line.set_ydata(f.evaluate(nodes))
    d2Psi_line.set_ydata(d2Psi)
    E_line.set_ydata(R)
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    ax3.relim()
    ax3.autoscale_view()
    ax4.relim()
    ax4.autoscale_view()
    plt.draw()

plt.show()
