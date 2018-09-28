#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from sklearn.linear_model import LinearRegression
from numba import jit
from sklearn.metrics import r2_score
from sklearn import linear_model

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

def OLS(X, z, n):
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(z))
    znew = X.dot(beta)
    plutt = znew.reshape((n,n))
    return plutt, beta

def ridge(X, z, n, lambd):
    beta = np.linalg.inv(X.T.dot(X)+lambd*np.identity(X.shape[1])).dot(X.T.dot(z))
    znew = X.dot(beta)
    plutt = znew.reshape((n,n))
    return plutt, beta

def FRANK(x, y):
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

z_plot = OLS(X, z, n)[0]
betas = OLS(X, z, n)[1]

temp = z_plot.reshape((n**2, 1))
true = FRANK(x,y).reshape((n**2, 1))

MSE = sum((true-temp)**2)/(len(true))
R2  = 1-(np.sum((true - temp)**2)/np.sum((true-np.mean(true))**2))

def KfoldCrossVal(dataset, dataset2, Numbfold):
    indices = np.arange(len(dataset[:, 0]))
    random_indices = np.random.choice(indices, size = len(dataset[:, 0]), replace = False)
    interval = int(len(dataset[:, 0])/Numbfold)
    datasetsplit = []
    dataset2split = []
    for k in range(Numbfold):
        datasetsplit.append(dataset[random_indices[interval*k : interval*(k + 1)]]) 
        dataset2split.append(dataset2[random_indices[interval*k : interval*(k + 1)]])
    return np.asarray(datasetsplit), np.asarray(dataset2split) 

Numbfolds = 5
X_Split, z_Split = KfoldCrossVal(X, z, Numbfolds)

def OLStrain(X, z):
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(z))
    return beta

def ridgetrain(X, z, lambd):
    beta = np.linalg.inv(X.T.dot(X)+lambd*np.identity(X.shape[1])).dot(X.T.dot(z))
    return beta


#predLassoplot = predLasso.reshape((n, n))
#print (predLasso.shape, z.shape, xnew.shape, predLassoplot.shape)


R2_ScoresOLS = np.zeros(Numbfolds)
R2_ScoresRidge = np.zeros(Numbfolds)
R2_ScoresLasso = np.zeros(Numbfolds)

for i in range(Numbfolds):
    XTrainSets = np.delete(X_Split, i, 0)
    XTestSets = X_Split[i]
    zTrainSets = np.delete(z_Split, i, 0)
    zTestSets = z_Split[i]
    XTrainSets = np.vstack(XTrainSets)
    zTrainSets = np.vstack(zTrainSets)
    #print (XTrainSets.shape, zTrainSets.shape)
    #print(XTestSets.shape, zTestSets.shape)

    betaTrainOLS = OLStrain(XTrainSets, zTrainSets)
    zTestedOLS = np.dot(XTestSets, betaTrainOLS)
    R2_ScoresOLS[i] = r2_score(zTestSets, zTestedOLS)

    betaTrainRidge = ridgetrain(XTrainSets, zTrainSets, 0.1)
    zTestedRidge = np.dot(XTestSets, betaTrainRidge)
    R2_ScoresRidge[i] = r2_score(zTestSets, zTestedRidge)

    lasso = linear_model.Lasso(alpha = 0.001)
    lasso.fit(XTrainSets, zTrainSets)
    predLasso = lasso.predict(XTestSets)
    R2_ScoresLasso[i] = r2_score(zTestSets, predLasso)

print ('CROSS VALIDATION RESAMPLE TECHNIQUE')
print ('--------------------------------------------------------------------')
print ('R2 scores for resampled data with OLS:')
print (R2_ScoresOLS)
print ('--------------------------------------------------------------------')

print ('--------------------------------------------------------------------')
print ('R2 scores for resampled data with Ridge:')
print (R2_ScoresRidge)
print ('--------------------------------------------------------------------')

print ('--------------------------------------------------------------------')
print ('R2 scores for resampled data with Lasso:')
print (R2_ScoresLasso)
print ('--------------------------------------------------------------------')

#X_train = np.r_[X_Split[0] , X_Split[1], X_Split[2], X_Split[3]]
#X_test = X_Split[4]
#z_train = np.r_[z_Split[0] , z_Split[1], z_Split[2], z_Split[3]]
#z_test = z_Split[4]
#print(X_train.shape, z_train.shape)
#print(X_test.shape, z_test.shape)
#betaTRAIN = OLStrain(X_train, z_train)
#ztested = np.dot(X_test, betaTRAIN)
#R2_Scores[i] = r2_score(z_test, ztested)

#May be useful 
#indices = np.arange(len(x1d))
#random_indices = np.random.choice(indices, size = len(x1d), replace = False)
#interval = int(len(x1d)/5)
#k1 = random_indices[0 : interval] 
#k2 = random_indices[interval : interval*2]
#k3 = random_indices[interval*2 : interval*3]
#k4 = random_indices[interval*3 : interval*4]
#k5 = random_indices[interval*4 : interval*5]
#
#X_train1 = np.r_[X[k1], X[k2], X[k3], X[k4]]
#z_train1 = np.r_[z[k1], z[k2], z[k3], z[k4]]
#
#X_test1 = X[k5]
#z_test1 = z[k5]
#
#betaTRAIN1 = OLStrain(X_train1, z_train1)
#ztested1 = np.dot(X_test1, betaTRAIN1)
#Guuutaher = r2_score(z_test1, ztested1)
#print (ztested, ztested1)
#print ('guuut', Guuutaher)


#print(max(z-temp))

#print('--------------------------------')
#print('R2 score for OLS is using sklearn:%0.3f'% r2_score(true, temp))
#print('R2 score for OLS is using analytic:%0.3f'% R2)
#print('MSE for OLS is using analytic calc %0.3f:'% MSE)
#print('--------------------------------')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib import cm
fig = plt.figure()
ax = fig.gca(projection="3d")
# Plot the surface.
surf = ax.plot_surface(x,y,z_plot, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
