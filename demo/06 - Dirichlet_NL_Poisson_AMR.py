import math as m
import numpy as np

from matplotlib import pyplot as plt

from BDPoisson1D import dirichlet_non_linear_poisson_solver_amr
from BDFunction1D import Function
from BDFunction1D.Functional import Functional


class TestFunction(Function):
    """
    Some known differentiable function
    """
    def evaluate_point(self, x):
        return m.exp(-x * 3)


class TestFunctional(Functional):
    def __init__(self, Nd, kT, f):
        super(TestFunctional, self).__init__(f)
        self.Nd = Nd
        self.kT = kT

    def evaluate_point(self, x):
        return self.Nd(x) * (1 - (m.exp(-self.f.evaluate_point(x) / self.kT)))


class TestFunctionalDf(Functional):
    def __init__(self, Nd, kT, f):
        super(TestFunctionalDf, self).__init__(f)
        self.Nd = Nd
        self.kT = kT

    def evaluate_point(self, x):
        return self.Nd(x) / self.kT * m.exp(-self.f.evaluate_point(x) / self.kT)


Nd = lambda x: np.ones_like(x)
kT = 1 / 20

Psi = TestFunction()
f = TestFunctional(Nd, kT, Psi)
dfdPsi = TestFunctionalDf(Nd, kT, Psi)

start = 0.0
stop = 5.0
step = 0.5
step_plot = step / 100
bc1 = 1.0
bc2 = 0.0

solution = dirichlet_non_linear_poisson_solver_amr(start, stop, step, Psi, f, dfdPsi, bc1, bc2,
                                                   max_iter=1000, residual_threshold=1.5e-3,
                                                   int_residual_threshold=1.5e-4,
                                                   max_level=20, mesh_refinement_threshold=1e-7)
print('checking BC. SOL:', solution.evaluate_point(start), solution.evaluate_point(stop), 'REF:', bc1, bc2)

fig, (ax1, ax2) = plt.subplots(2, sharex=True)

nodes = np.linspace(start, stop, num=int((stop-start)/step_plot+1))

ax1.plot(nodes, np.asarray(solution.evaluate(nodes)), '-')
ax2.plot(nodes, np.asarray(solution.error(nodes)), '-')
plt.show()
