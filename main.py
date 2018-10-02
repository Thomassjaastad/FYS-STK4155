#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from sklearn.linear_model import LinearRegression
from numba import jit
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import linear_model
import Reg
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

n = 200
deg = 5
x = np.sort((np.random.rand(n)))
y = np.sort((np.random.rand(n)))

x, y = np.meshgrid(x,y)

x1d = x.reshape((n**2, 1))
y1d = y.reshape((n**2, 1))

error = 0.1*np.random.randn(n**2, 1)

z = FRANK(x1d,y1d) + error

X = Reg.polynomial_this(x1d, y1d, deg)

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

Numbfolds = 30
X_Split, z_Split = KfoldCrossVal(X, z, Numbfolds)

lambdas = [10e-4 , 10e-3, 10e-2, 10e-1, 10e0, 10e1, 10e2, 10e3, 10e4]

def R2lambda(dataset1, dataset2, lambdas):
    Franklamb = Reg.regression(dataset1, dataset2, n)
    R2ridge = []
    R2lasso = []
    R2OLS = []
    for lambd in lambdas:
        betasR = Franklamb.ridge(lambd)
        zRidge = np.dot(X, betasR)
        R2ridge.append(r2_score(z, zRidge))

        betasL = Franklamb.lasso(lambd)
        zLasso = np.dot(X, betasL)
        R2lasso.append(r2_score(z, zLasso))

        betasOLS = Franklamb.OLS()
        zOLS = np.dot(X, betasOLS)
        R2OLS.append(r2_score(z, zOLS))
    return R2OLS, R2ridge, R2lasso

R2OLSTrain, R2ridgeTrain, R2lassoTrain = R2lambda(np.vstack(X_Split[0 :,:]), np.vstack(z_Split[0 :, :]), lambdas)
R2OLSTest, R2ridgeTest, R2lassoTest = R2lambda(X_Split[0 , :], z_Split[0 , :], lambdas)

###plt.plot(np.log10(lambdas),R2ridgeTest, '-' ,  label = 'Ridge Test', color = 'red')
##plt.plot(np.log10(lambdas),R2OLSTest, '-' ,  label = 'OLS Test', color = 'blue')
#plt.plot(np.log10(lambdas), R2lassoTest, '-' ,  label = 'Lasso Test', color = 'green')
#plt.plot(np.log10(lambdas),R2ridgeTrain, '--' ,  label = 'Ridge Train', color = 'red')
#plt.plot(np.log10(lambdas),R2OLSTrain, '--' ,  label = 'OLS Train', color = 'blue')
#plt.plot(np.log10(lambdas), R2lassoTrain, '--' ,  label = 'Lasso Train', color = 'green')
#plt.xlabel(r'$log_{10}(\lambda)$', fontsize = 16)
#plt.ylabel(r'R2', fontsize = 16)
#plt.legend()  

beta_meanOLS = np.zeros((len(X[0, :]), 1))
beta_meanRidge = np.zeros((len(X[0, :]), 1))
beta_meanLasso = np.zeros((len(X[0, :]), 1))

for i in range(Numbfolds):
    XTrainSets = np.delete(X_Split, i, 0)
    XTestSets = X_Split[i]
    zTrainSets = np.delete(z_Split, i, 0)
    zTestSets = z_Split[i]
    XTrainSets = np.vstack(XTrainSets)
    zTrainSets = np.vstack(zTrainSets)
    Frank = Reg.regression(XTrainSets, zTrainSets, n)

    betaTrainOLS = Frank.OLS()
    beta_meanOLS += betaTrainOLS
    
    betaTrainRidge = Frank.ridge(0.1)
    beta_meanRidge += betaTrainRidge

    betaTrainLasso = Frank.lasso(0.0001)
    beta_meanLasso += betaTrainLasso  


#OLS evaluated on Franke function
beta_meanOLS /= Numbfolds
zOLSmean = np.dot(X, beta_meanOLS)
R2ScoreOLS = r2_score(z, zOLSmean)
MSEOLS = Frank.MSE()[1]
BiasOLS = Frank.Bias()[1]
VarOLS = Frank.Variance()[1]
print(MSEOLS, BiasOLS, VarOLS)

#Ridge evaluated on Franke function
beta_meanRidge /= Numbfolds
zRidgemean = np.dot(X, beta_meanRidge)
R2ScoreRidge = r2_score(z, zRidgemean)
BiasRidge = Frank.Bias()[1]
MSERidge = Frank.MSE()[1]
VarRidge = Frank.Variance()[1]

#Lasso evaluated Franke function
beta_meanLasso /= Numbfolds
zLassomean = np.dot(X, beta_meanLasso)
R2ScoreLasso = r2_score(z, zLassomean)
BiasLasso = Frank.Bias()[1]
MSELasso = Frank.MSE()[1]
VarLasso = Frank.Variance()[1]

Numbdeg = 30
Variterate = np.zeros(Numbdeg)
Biasiterate = np.zeros(Numbdeg)
MSEiterate = np.zeros(Numbdeg)
for k in range(Numbdeg):
    Xiterate = Reg.polynomial_this(x1d, y1d, k)
    ziterate = FRANK(x1d, y1d) + error

    Frankiterate = Reg.regression(Xiterate, z, n)
    betas = Frankiterate.OLS()
    MSEiterate[k] = Frankiterate.MSE()[0]  
    Biasiterate[k] = Frankiterate.Bias()[0]
    Variterate[k] = Frankiterate.Variance()[0] 
    #print(Variterate[k], Biasiterate[k])
plt.plot(np.linspace(0, Numbdeg-1, Numbdeg), Variterate, '*', label = 'Var')
plt.plot(np.linspace(0, Numbdeg-1, Numbdeg), Biasiterate, '--', label = 'Bias**2')
plt.plot(np.linspace(0, Numbdeg-1, Numbdeg), MSEiterate, 'o', label = 'MSE')
plt.legend()
plt.show()
exit()

#Plotting
fig = plt.figure()
ax = fig.gca(projection= "3d")


#Plot the surface.
Frank = ax.plot_surface(x, y, z.reshape((n, n)), cmap=cm.coolwarm, linewidth=0, antialiased=False)

#Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
#Add a color bar which maps values to colors.
fig.colorbar(Frank, shrink=0.5, aspect=5)
#plt.show()