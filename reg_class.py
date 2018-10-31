import numpy as np
from sklearn import linear_model
from numba import jit
#from sklearn.linear_model import LinearRegression

@jit
def polynomial_this(x,y,n):
    if x.shape[-1] != 1 and x.shape[0] !=1:
        x = x.ravel()
        y = y.ravel()
    X = np.c_[np.ones(len(x))]
    for i in range(1,n+1):
        X = np.c_[X,x**(i)]
        for j in range(i-1,0,-1):
            X = np.c_[X,(x**(j))*(y**(i-j))]  
        X = np.c_[X,y**(i)]
    return X

def bias(true, pred):
    bias = np.mean((true - np.mean(pred))**2)
    return bias

    
def MSE(true, pred):
    MSE = sum((true - pred)**2)/(len(true))
    return MSE
    
def R2(true, pred):
    R2 = 1-(np.sum((true - pred)**2)/np.sum((true-np.mean(pred))**2))
    return R2

def variance(pred):
    var = np.mean(np.var(pred))
    return var

def KfoldCrossVal(dataset, dataset2, Numbfold):
    """
    Takes in two coupled datasets and returns them splitted into k-matching 
    """
    indices = np.arange(len(dataset[:, 0]))
    random_indices = np.random.choice(indices, size = len(dataset[:, 0]), replace = False)
    interval = int(len(dataset[:, 0])/Numbfold)
    datasetsplit = []
    dataset2split = []
    for k in range(Numbfold):
        datasetsplit.append(dataset[random_indices[interval*k : interval*(k + 1)]]) 
        dataset2split.append(dataset2[random_indices[interval*k : interval*(k + 1)]])

    return np.asarray(datasetsplit), np.asarray(dataset2split) 


class regression:
    def __init__(self,X,z):
        self.z = z
        self.X = X
        
    @jit    
    def ridge(self,lambd):
        X = self.X
        beta = np.linalg.inv(X.T.dot(X)+lambd*np.identity(X.shape[1])).dot(X.T.dot(self.z))
        self.znew = X.dot(beta)

        self.beta = beta
        return beta#plutt
    
    @jit
    def OLS(self):
        X = self.X
        beta = np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(self.z))
        self.znew = X.dot(beta)
        return beta
    
    @jit
    def lasso(self, lambd):
        lasso = linear_model.Lasso(alpha = lambd,fit_intercept = False)
        lasso.fit(self.X, self.z)
        beta = lasso.coef_
        self.znew = self.X.dot(beta)
        return beta

         
    def beata_variance(self):
        sigma2 = (1./(len(self.z)-self.X.shape[1]-1))*sum((self.z-self.znew)**2)
        covar = np.linalg.inv(self.X.T.dot(self.X))*sigma2
        var = np.diagonal(covar)
        return beta_var
    



    def plot(self):
        plutt = self.znew.reshape((self.n,self.n))
        x = self.x.reshape((self.n,self.n))
        y = self.y.reshape((self.n,self.n))
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        import matplotlib.pyplot as plt
        from matplotlib import cm
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        # Plot the surface.
        surf = ax.plot_surface(x,y,plutt, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        # Customize the z axis.
        #ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        return plutt
    
    def MSE(self):
        MSE = np.mean((self.z-self.znew)**2)
        return MSE
    
    def R2(self):
        self.R2 = 1-(np.sum((self.z - self.znew)**2)/np.sum((self.z-np.mean(self.z))**2))
        return self.R2


def FRANK(x, y):
    """
    Frankie function for testing the class
    """
    term1 = 0.75*np.exp(-(0.25*(9*x - 2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x + 1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4    
    


if __name__== "__main__" :
    def test_reg():
        a = 3       
    
