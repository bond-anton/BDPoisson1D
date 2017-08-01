from __future__ import division, print_function
import numpy as np
from matplotlib import pyplot as plt

from BDPoisson1D import dirichlet_non_linear_poisson_solver_recurrent_mesh
from BDMesh import Mesh1DUniform, TreeMesh1DUniform

Nd = lambda x: np.ones_like(x)
kT = 1 / 20


def f(x, Psi):
    return Nd(x) * (1 - (np.exp(-Psi(x) / kT)))


def dfdDPsi(x, Psi):
    return Nd(x) / kT * np.exp(-Psi(x) / kT)


Psi = lambda x: np.exp(-x * 3)


bc1 = 1
bc2 = 0

root_mesh = Mesh1DUniform(0.0, 10.0, bc1, bc2, 0.2)

root_mesh, Psi = dirichlet_non_linear_poisson_solver_recurrent_mesh(root_mesh, Psi, f, dfdDPsi, max_iter=1000,
                                                                    threshold=1e-6)

mesh_refinement_threshold = 1e-7
idxs = np.where(abs(root_mesh.residual) > mesh_refinement_threshold)

dPsi = np.gradient(root_mesh.solution, root_mesh.physical_nodes, edge_order=2)
d2Psi = np.gradient(dPsi, root_mesh.physical_nodes, edge_order=2)

_, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
ax1.plot(root_mesh.physical_nodes, root_mesh.solution)
ax1.plot(root_mesh.physical_nodes[idxs], root_mesh.solution[idxs], 'r-o')
ax2.plot(root_mesh.physical_nodes, root_mesh.residual)
ax2.plot(root_mesh.physical_nodes[idxs], root_mesh.residual[idxs], 'r-o')
ax3.plot(root_mesh.physical_nodes, f(root_mesh.physical_nodes, Psi))
ax3.plot(root_mesh.physical_nodes, d2Psi)
plt.show()
