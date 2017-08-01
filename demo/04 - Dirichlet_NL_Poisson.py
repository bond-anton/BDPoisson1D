from __future__ import division, print_function
import numpy as np
from matplotlib import pyplot as plt

from BDPoisson1D import dirichlet_non_linear_poisson_solver

Nd = lambda x: np.ones_like(x)
kT = 1 / 20


def f(x, Psi):
    return Nd(x) * (1 - (np.exp(-Psi(x) / kT)))


def dfdDPsi(x, Psi):
    return Nd(x) / kT * np.exp(-Psi(x) / kT)


Psi = lambda x: np.exp(-x * 3)

nodes = np.linspace(0., 4., num=21, endpoint=True, dtype=np.float)
bc1 = 1
bc2 = 0

print(nodes, nodes.size, nodes.dtype)

DPsi = np.zeros_like(nodes)
E = np.zeros_like(nodes)
_, (ax1, ax2, ax3, ax4) = plt.subplots(4)
ax1.set_autoscaley_on(True)
ax2.set_autoscaley_on(True)
ax3.set_autoscaley_on(True)
ax4.set_autoscaley_on(True)
Psi_line, = ax1.plot(nodes, Psi(nodes))
DPsi_line, = ax2.plot(nodes, DPsi)
f_line, = ax3.plot(nodes, f(nodes, Psi))
E_line, = ax4.plot(nodes, E)
print(Psi(nodes), Psi(nodes).size, Psi(nodes).dtype)

dPsi = np.gradient(Psi(nodes), nodes, edge_order=2)
print(dPsi, dPsi.size)
d2Psi = np.gradient(dPsi, nodes, edge_order=2)
print(d2Psi, d2Psi.size)
d2Psi_line, = ax3.plot(nodes, d2Psi)


plt.draw()


for i in range(100):
    print(i + 1)
    Psi, DPsi, R = dirichlet_non_linear_poisson_solver(nodes, Psi, f, dfdDPsi, bc1=1, bc2=0, j=1)
    dPsi = np.gradient(Psi(nodes), nodes, edge_order=2)
    d2Psi = np.gradient(dPsi, nodes, edge_order=2)
    Psi_line.set_ydata(Psi(nodes))
    DPsi_line.set_ydata(DPsi)
    f_line.set_ydata(f(nodes, Psi))
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
