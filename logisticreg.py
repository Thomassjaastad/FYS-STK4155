import numpy as np
import time
from sklearn.linear_model import LogisticRegression
import warnings
import classymlp
import matplotlib.pyplot as plt

#Comment this to turn on warnings
warnings.filterwarnings('ignore')

np.random.seed(1) # shuffle random seed generator

# Ising model parameters
L = 40 # linear system size
J = -1.0 # Ising interaction
T = np.linspace(0.25,4.0,16) # set of temperatures
T_c = 2.26 # Onsager critical temperature in the TD limit
##### prepare training and test data sets

import pickle,os
from sklearn.model_selection import train_test_split

###### define ML parameters
num_classes = 2
train_to_test_ratio = 0.8 # training samples

# path to data directory
path_to_data=os.path.expanduser

# load data
file_name = "Ising2DFM_reSample_L40_T=All.pkl"   # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
data = pickle.load(open(file_name,'rb'))         # pickle reads the file and returns the Python object (1D array, compressed bits)
data = np.unpackbits(data).reshape(-1, 1600)     # Decompress array and reshape for convenience
data=data.astype('int')
data[np.where(data==0)]=-1                       # map 0 state to -1 (Ising variable can take values +/-1)

file_name = "Ising2DFM_reSample_L40_T=All_labels.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
labels = pickle.load(open(file_name,'rb'))            # pickle reads the file and returns the Python object (here just a 1D array with the binary labels)

# divide data into ordered, critical and disordered
X_ordered=data[:70000,:]
Y_ordered=labels[:70000]

X_disordered=data[100000:,:]
Y_disordered=labels[100000:]

del data,labels

# define training and test data sets
X = np.concatenate((X_ordered, X_disordered))
Y = np.concatenate((Y_ordered, Y_disordered))
X = np.c_[np.ones(X.shape[0]),X]


# Validationset
validationInputs = 20000
validationset = X[:validationInputs]
validationtargets = Y[:validationInputs]
X = np.delete(X, np.s_[:validationInputs], axis=0)
Y = np.delete(Y, np.s_[:validationInputs], axis=0)

# pick random data points from ordered and disordered states 
# to create the training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = train_to_test_ratio)

random_index = np.random.randint(X_train.shape[0])
indices = np.arange(len(X_train[:, 0]))
random_indices = np.random.choice(indices, size = len(X_train[:, 0]), replace = False)


# full data set
X_train = X_train[random_indices]
Y_train = Y_train[random_indices]

# Logistic regression implementation
weights = np.random.randn(*(X_train.shape[1], 1))*0.001
eta_const = 0.1

def Sigmoid(x):
    sig = 1/(1 + np.exp(-x))
    return sig

def learn_me(t):
    t0, t1 = 5, 50
    return t0/(t + t1)

# Set parameters
iterations = 300
numb_batches = 50
X_batches = np.split(X_train, numb_batches, axis = 0)
X_batches = np.array(X_batches)
Y_batches = np.split(Y_train, numb_batches, axis = 0)
Y_batches = np.array(Y_batches)

# logistic regression
lmbda = np.logspace(-5,4,10)

def Accuracy(test, testtargets):
    accuracy = 0
    Y_testpred = np.dot(test, weights)
    sig_y = Sigmoid(Y_testpred)
    testtargets = testtargets[:, np.newaxis] 
    for i in range(test.shape[0]):  
        if sig_y[i] < 0.5:
            sig_y[i] = 0
        else:
            sig_y[i] = 1  
        if sig_y[i] == testtargets[i]:
            accuracy +=1
    return accuracy/test.shape[0]*100

sum_error = np.zeros(iterations)
accOLSTest = []
accOLSTrain = []
#start = time.time()
for lmbd in lmbda:
    for iters in range(iterations):
        sum_gradi = 0
        for i in range(numb_batches):
            h = np.dot(X_batches[i], weights)
            Y_pred = Sigmoid(h)
            Y_batches_new = Y_batches[i][:, np.newaxis]
            output_error = Y_pred - Y_batches_new
            gradient = np.dot(X_batches[i].T, output_error) #+ lmbd*np.linalg.norm(weights, axis=0)**2
            sum_gradi += np.sum(gradient)
            eta = learn_me(iters*i)
            weights = weights - eta*gradient 
        sum_error[iters] = sum_gradi
    scoreTest = Accuracy(X_test, Y_test)
    scoreTrain = Accuracy(X_train, Y_train)
    accOLSTest.append(scoreTest)
    accOLSTrain.append(scoreTrain)
    print('-----------------------------------------------')
    print('lambda =', lmbd)
    print('Accuracy score on test data for logistic regression is:', scoreTest)
    print('Accuracy score on training data for logistic regression is:', scoreTrain)
    print('-----------------------------------------------')
    #end = time.time()
    #print("tid", end - start)

#print(accOLSTest, accOLSTrain, lmbda)

plt.semilogx(lmbda, accOLSTest, '*-r', label = 'Test')
plt.semilogx(lmbda, accOLSTrain, '*--r', label = 'Train')
plt.xlabel(r'$\lambda$', fontsize = 18)
plt.ylabel(r'Accuracy', fontsize = 18)
plt.legend()
plt.show()
# Using sklearn logistic regressor
#logreg = LogisticRegression(C=1.0, random_state=1,verbose=0,max_iter=1E3,tol=1E-5)
#logreg.fit(X_train, Y_train)
#test_accuracy = logreg.score(X_test, Y_test)
#print('-----------------------------------------------')
#print('Accuracy score using sklearn logistic regression is:', test_accuracy*100)
#print('-----------------------------------------------')

"""
hidden_nodes = 5
Y_train = Y_train[:, np.newaxis]
neural = classymlp.mlp(X_train, Y_train, hidden_nodes)
error_cross, epochs = neural.earlystopping(X_train, Y_train, validationset, validationtargets)
error_cross = np.array(error_cross)
neural.accuracy_score(X_test, Y_test)


##### plot a few Ising states
#matplotlib inline

from mpl_toolkits.axes_grid1 import make_axes_locatable

#Classfier validation error plot
epochs = 50
epoch = np.linspace(1, epochs, epochs)

plt.plot(epoch, error_cross, label= 'Validation set')
plt.xlabel(r'Number of epochs', fontsize = 18)
plt.ylabel(r'Cross entropy error', fontsize = 18)
plt.legend()
plt.show()
"""
"""
# set colourbar map
cmap_args=dict(cmap='plasma_r')

# plot states
fig, axarr = plt.subplots(nrows=1, ncols=2)

axarr[0].imshow(X_ordered[20001].reshape(L,L),**cmap_args)
axarr[0].set_title('$\\mathrm{ordered\\ phase}$',fontsize=16)
axarr[0].tick_params(labelsize=16)

#axarr[1].imshow(X_critical[10001].reshape(L,L),**cmap_args)
#axarr[1].set_title('$\\mathrm{critical\\ region}$',fontsize=16)
#axarr[1].tick_params(labelsize=16)

axarr[1].imshow(X_disordered[50001].reshape(L,L),**cmap_args)
axarr[1].set_title('$\\mathrm{disordered\\ phase}$',fontsize=16)
axarr[1].tick_params(labelsize=16)

#fig.subplots_adjust(right=2.0)

plt.show()
"""
