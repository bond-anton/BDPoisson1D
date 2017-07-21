from __future__ import division, print_function
import numpy as np
from matplotlib import pyplot as plt

from BDPoisson1D import dirichlet_poisson_solver


def Y(x):
    # return x**2
    return np.cos(2 * np.pi * x)


# def f(x):
#    return -(2*np.pi)**2 * np.cos(2*np.pi * x)

def f(x):
    y = Y(x)
    dx = np.gradient(x)
    dy = np.gradient(y, dx, edge_order=2)
    d2y = np.gradient(dy, dx, edge_order=2)
    return d2y


nodes = np.linspace(-1.0, 1.0, num=11, endpoint=True)
bc1 = Y(nodes[0])
bc2 = Y(nodes[-1])
Y_sol, d2Y_sol, R = dirichlet_poisson_solver(nodes, f, bc1, bc2)

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(nodes, Y(nodes), 'radius-')
ax1.plot(nodes, Y_sol, 'b-')
# ax1.plot(nodes, Y_sol-R, 'g-o')
# ax1.plot(nodes, Y_sol+R, 'g-o')
ax2.plot(nodes, R, 'g-o')
# ax2.plot(nodes, f(nodes), 'r-')
# ax2.plot(nodes, d2Y_sol, 'b-')
plt.show()
