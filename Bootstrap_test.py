#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from sklearn.linear_model import LinearRegression
from numba import jit

@jit
def polynomial_this(x,y,n):
    X = np.c_[np.ones(len(x))]
    for i in range(1,n+1):
        X = np.c_[X,x**(i)]
        for j in range(i-1,0,-1):
            X = np.c_[X,(x**(j))*(y**(i-j))]  
        X = np.c_[X,y**(i)]
    return X
@jit

def OLS(x,y,z,n,deg):
    X = polynomial_this(x,y,deg)
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(z))
    znew = X.dot(beta)
    plutt = znew.reshape((n,n))
    return plutt, beta


def ridge(x,y,z,n,deg,lambd):
    X = polynomial_this(x,y,deg)
    beta = np.linalg.inv(X.T.dot(X)+lambd*np.identity(X.shape[1])).dot(X.T.dot(z))
    znew = X.dot(beta)
    plutt = znew.reshape((n,n))
    return plutt, beta

def FRANK(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1+term2+term3+term4

#oppg 1
n = 100
deg = 5
x = np.sort((np.random.rand(n)))
y = np.sort((np.random.rand(n)))

x, y = np.meshgrid(x,y)



x1d = x.reshape((n**2,1))
y1d = y.reshape((n**2,1))

X = polynomial_this(x1d, y1d, deg)
z = FRANK(x1d,y1d) #+5*np.random.randn(n*n,1)

z_plot = OLS(x1d,y1d,z,n,deg)[0]
temp = z_plot.reshape((n**2,1))
true = FRANK(x,y).reshape((n**2,1))

MSE = sum((true-temp)**2)/(len(true))
R2  = 1-(np.sum((true - temp)**2)/np.sum((true-np.mean(true))**2))

indices = np.arange(len(x1d))
random_indices = np.random.choice(indices, size = len(x1d), replace = False)
interval = int(len(x1d)/5)
k1 = random_indices[0 : interval] 
k2 = random_indices[interval : interval*2]
k3 = random_indices[interval*2 : interval*3]
k4 = random_indices[interval*3 : interval*4]
k5 = random_indices[interval*4 : interval*5]

X_test1 = X[k1] 
#z_test1 = temp[k1]
print (X_test1.shape, temp[k1].shape)

#def k_fold_cross(x, z):
#	x_test = 
#	x_train = 


#def Bootstrap(data, Numbsamples):
	#Remember to take square root of answers zlasso = (n**2, 1) 
#	mu = np.zeros(Numbsamples)
#	for i in range(Numbsamples):
#		estimate = np.random.choice(data[:, 0], Numbsamples)
#		mu[i] = np.mean(estimate)
#	return mu

#Bootstrap(temp, 300)

from sklearn.metrics import r2_score

#print(max(z-temp))

print('--------------------------------')
print('R2 score for OLS is using sklearn:%0.3f'% r2_score(true, temp))
print('R2 score for OLS is using analytic:%0.3f'% R2)
print('MSE for OLS is using analytic calc %0.3f:'% MSE)
print('--------------------------------')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib import cm
fig = plt.figure()
ax = fig.gca(projection="3d")
# Plot the surface.
surf = ax.plot_surface(x,y,z_plot, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
