#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from sklearn.linear_model import LinearRegression
from numba import jit
from sklearn import linear_model

@jit
def polynomial_this(x, y, deg):
    X = np.c_[np.ones(len(x))]
    for i in range(1, deg + 1):
        X = np.c_[X, x**(i)]
        for j in range(i - 1, 0, -1):
            X = np.c_[X, (x**(j))*(y**(i-j))]  
        X = np.c_[X, y**(i)]
    return X

class regression:
    def __init__(self, X, z, n):
        self.X = X
        self.z = z
        self.n = n
        #self.deg = deg
        
    @jit    
    def ridge(self, lambd):
        X = self.X
        beta = np.linalg.inv(X.T.dot(X)+lambd*np.identity(X.shape[1])).dot(X.T.dot(self.z))
        self.znew = X.dot(beta)
        #self.beta = beta
        return beta
    
    @jit
    def OLS(self):
        X = self.X
        beta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(self.z))
        self.znew = X.dot(beta)
        return beta
    
    @jit
    def lasso(self, alphain):
        lasso = linear_model.Lasso(alpha = alphain, fit_intercept=False)
        lasso.fit(self.X, self.z)
        beta = lasso.coef_
        beta = beta[:, np.newaxis]
        self.znew = self.X.dot(beta)
        return beta

    def VarianceBeta(self):
        omegaSQ = sum(self.z - self.znew)**2/(len(self.z) - len(self.z[0,:]) - 1)
        var = np.linalg.inv(self.X.T.dot(self.X))*omegaSQ
        varbeta = var.diagonal()
        return varbeta
        
    def Variance(self):
        Var1 =  np.mean( np.var(self.znew, axis = 1, keepdims = True))
        Var = np.mean( np.var(self.znew))
        return Var, Var1

    def plot(self):
        plutt = self.znew.reshape((self.n, self.n))
        return plutt
    
    def MSE(self):
        MSE1 = np.mean( np.mean((self.z - self.znew)**2, axis = 1, keepdims = True ))
        MSE = np.mean( np.mean((self.z - self.znew)**2))
        return MSE, MSE1
    
    def R2(self):
        R2 = 1-(np.sum((self.z - self.znew)**2)/np.sum((self.z - np.mean(self.z))**2))
        return R2
    
    def Bias(self):
        Bias1 = np.mean( (self.z - np.mean(self.znew, axis = 1, keepdims = True) )**2)
        Bias = np.mean( (self.z - np.mean(self.znew))**2)
        return Bias, Bias1

    def error(self):
        error = (np.sum(self.z - np.mean(self.znew))*(np.mean(self.znew) - self.znew)*2/len(self.z))
        return error
"""
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# Load the terrain
terrainfull = np.array(imread("SRTM_data_Norway_1.tif"))
terrain1 = terrainfull[1000:2000,800:1800]
tx = np.linspace(0,1,terrain1.shape[0])
ty = np.linspace(0,1,terrain1.shape[1])
tx,ty = np.meshgrid(tx,ty)
tx1d = tx.ravel()
ty1d = ty.ravel()
terrain11d=terrain1.ravel()
print(terrain1.shape,tx.shape,ty.shape)
deg = 20
terreng = regression(tx1d,ty1d,terrain11d,5,deg)
terreng_beta = terreng.ridge(0.001)
print(terreng_beta)
TX = polynomial_this(tx1d,ty1d,deg)
terengplutt = TX.dot(terreng_beta)
terengplutt = terengplutt.reshape((terrain1.shape[0],terrain1.shape[1]))
print(terreng.R2(),"GUNNNNNNNNARERBEST")
# Show the terrain
from matplotlib.ticker import LinearLocator, FormatStrFormatter
fig = plt.figure()
ax = fig.gca(projection="3d")
# Plot the surface.
surf = ax.plot_surface(-tx,-ty,terrain1, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
# Customize the z axis.
#ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.hold("on")
#plt.show()
plt.figure()
plt.title("Terrain_over_Norway_1")
plt.imshow(terrain1, cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
plt.hold("on")
#plt.show()
#oppg 1"""
#n = 100
#deg = 5
#x = np.sort((np.random.rand(n)))
#y = np.sort((np.random.rand(n)))
#
#x,y = np.meshgrid(x,y)
#
#x1d = x.reshape((n**2,1))
#y1d = y.reshape((n**2,1))
#
#
#z = FRANK(x1d,y1d) #+5*np.random.randn(n*n,1)
#
#
#frankreg = regression(x1d,y1d,z,n,deg)
#betaols = frankreg.OLS()
#OLSR2 = frankreg.R2()

"""
ridge = regression(x1d,y1d,z,n,deg)
ridge_beta = ridge.ridge(0.01)
z_plot = ridge.plot()
temp = z_plot.reshape((n**2,1))
true = FRANK(x,y).reshape((n**2,1))"""

#MSE = sum((true-temp)**2)/(len(true))
#R2  = 1-(np.sum((true - temp)**2)/np.sum((true-np.mean(true))**2))

#from sklearn.metrics import r2_score

#print(max(z-temp))

#print(r2_score(z, frankreg.plot().ravel()),"sklearn")
"""
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib import cm
fig = plt.figure()
ax = fig.gca(projection="3d")
# Plot the surface.
surf = ax.plot_surface(-tx,-ty,terengplutt, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
# Customize the z axis.
#ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()"""