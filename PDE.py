import numpy as np
import matplotlib.pyplot as plt

# boundary conditions
L = 1
n = 1

m = 100
k = 30000
# timestep and rodstep
dx = L/(m + 1)
dt = 1/(k + 1)
def g(x):
    """returns array of initial values at t = 0""" 
    return np.sin(np.pi*x)


alpha = dt/dx**2 

# initial condition
u = np.zeros(m + 1)
unew = np.zeros((m + 1, k))
u[0], u[m], unew[0, :], unew[- 1, :] = 0, 0, 0, 0

for i in range(1, m):
    x = i*dx
    u[i] = g(x)
unew[:, 0] = u

# solver
for t in range(1, k):
    for i in range(1, m):
        unew[i, t] = alpha*unew[i - 1, t - 1] + (1 - 2*alpha)*unew[i, t - 1] + alpha*unew[i + 1, t - 1]

def U(x, t):
    """return analytic solution for diffusion equation"""
    lamb = np.pi*n/L
    return np.sin(x*lamb)*np.exp(-t*lamb**2)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
"""Saves each time iteration plot and runs as sequence"""

x = np.linspace(0, L, m + 1)
t = np.linspace(0, 1, k)
line, = plt.plot(x, U(x, 0), label = 'analytic')
line2, = plt.plot(x, unew[:, 0], label = 'numerical')
for i, time in enumerate(t[::100]):
    line.set_ydata(U(x, time)) # analytic    
    line2.set_ydata(unew[:, i*100])
    plt.draw()
    plt.title('Diffusion equation at t = %f' % time)
    plt.pause(0.0001)   
    plt.legend()
plt.show()
