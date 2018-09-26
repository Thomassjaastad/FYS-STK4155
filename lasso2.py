#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from numba import jit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib import cm


def FRANK(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x - 2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x + 1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

#x = (np.random.rand(n))  #True values
#y = (np.random.rand(n))
#x, y = np.meshgrid(x, y)

n = 400
deg = 5

#Need to be sorted in order to plot
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)

#Meshgrid for plotting 
xnew, ynew = np.meshgrid(x, y) 

zlasso = FRANK(xnew, ynew)
X = np.c_[np.ones((n, 1)), x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3, x**4, x**3*y, x**2*y**2, x*y**3, y**4, x**5, x**4*y, x**3*y**2, x**2*y**3, x*y**4, y**5]


X_train, X_test, zlasso_train, zlasso_test = train_test_split(X, zlasso, test_size = 0.4, random_state = 42)

XTest = xnew[X_test[: , 0].size, :]
YTest = ynew[X_test[: , 0].size, :]
zlassoTest = FRANK(XTest, YTest)

lasso = linear_model.Lasso(alpha = 0.1, fit_intercept = False)
lasso.fit(X_train, zlasso_train)
predLasso = lasso.predict(X_test)
#print ('LassoTrain MSE: %.2f' % mean_squared_error(zlassoTest, predLasso))
#print ('LassoTrain R2 score %.2f' % r2_score(zlassoTrain, predLasso))

#Plot the surface
fig = plt.figure()
ax = fig.gca(projection = "3d")
#Lasso = ax.plot_surface(xnew, ynew, predLasso, cmap = cm.coolwarm, linewidth=0, antialiased = False)
#Frank = ax.plot_surface(xnew, ynew, FRANK(xnew, ynew), cmap = cm.coolwarm, linewidth=0, antialiased = False)

#Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

#Add a color bar which maps values to colors.
#fig.colorbar(Lasso, shrink=0.5, aspect=5)
#fig.colorbar(Frank, shrink=0.5, aspect=5)
plt.show()