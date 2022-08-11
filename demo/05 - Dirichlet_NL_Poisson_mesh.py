import math as m
import numpy as np
from matplotlib import pyplot as plt

from BDPoisson1D import dirichlet_non_linear_poisson_solver_recurrent_mesh
from BDMesh import Mesh1DUniform
from BDFunction1D import Function
from BDFunction1D.Functional import Functional
from BDFunction1D.Differentiation import NumericGradient


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


bc1 = 1.0
bc2 = 0.0

root_mesh = Mesh1DUniform(0.0, 10.0, bc1, bc2, 0.2)

print('checking BC. TF:', Psi.evaluate_point(0), Psi.evaluate_point(10), 'REF:', bc1, bc2)
Psi = dirichlet_non_linear_poisson_solver_recurrent_mesh(root_mesh, Psi, f, dfdPsi, max_iter=1000, threshold=1e-6)
print('checking BC. SOL:', Psi.evaluate_point(0), Psi.evaluate_point(10), 'REF:', bc1, bc2)
# Psi = InterpolateFunction(root_mesh.physical_nodes, root_mesh.solution)

mesh_refinement_threshold = 1e-7
idxs = np.where(abs(np.asarray(root_mesh.residual)) > mesh_refinement_threshold)

dPsi = NumericGradient(Psi)
d2Psi = NumericGradient(dPsi)

_, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
ax1.plot(np.asarray(root_mesh.physical_nodes), np.asarray(Psi.evaluate(root_mesh.physical_nodes)))
ax1.plot(np.asarray(root_mesh.physical_nodes)[idxs], np.asarray(root_mesh.solution)[idxs], 'r-o')
ax2.plot(np.asarray(root_mesh.physical_nodes), np.asarray(root_mesh.residual))
ax2.plot(np.asarray(root_mesh.physical_nodes)[idxs], np.asarray(root_mesh.residual)[idxs], 'r-o')
ax3.plot(np.asarray(root_mesh.physical_nodes), np.asarray(f.evaluate(root_mesh.physical_nodes)))
ax3.plot(np.asarray(root_mesh.physical_nodes), np.asarray(d2Psi.evaluate(root_mesh.physical_nodes)))
plt.show()
