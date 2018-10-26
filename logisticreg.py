import numpy as np

import warnings
#Comment this to turn on warnings
warnings.filterwarnings('ignore')

np.random.seed(1) # shuffle random seed generator

# Ising model parameters
L=40 # linear system size
J=-1.0 # Ising interaction
T=np.linspace(0.25,4.0,16) # set of temperatures
T_c=2.26 # Onsager critical temperature in the TD limit
##### prepare training and test data sets

import pickle,os
from sklearn.model_selection import train_test_split

###### define ML parameters
num_classes=2
train_to_test_ratio=0.5 # training samples

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

#X_critical=data[70000:100000,:]
#Y_critical=labels[70000:100000]

X_disordered=data[100000:,:]
Y_disordered=labels[100000:]

del data,labels

# define training and test data sets
X=np.concatenate((X_ordered,X_disordered))
Y=np.concatenate((Y_ordered,Y_disordered))

# pick random data points from ordered and disordered states 
# to create the training and test sets
X_train,X_test,Y_train,Y_test=train_test_split(X, Y, train_size=train_to_test_ratio)

# full data set
#X=np.concatenate((X_critical,X))
#Y=np.concatenate((Y_critical,Y))

#print('X_train shape:', X_train.shape)
#print('Y_train shape:', Y_train.shape)
#print()
#print(X_train.shape, 'train samples')
#print(X_critical.shape[0], 'critical samples')
#print(X_test.shape[0], 'test samples')


# Logistic regression implementation
weights = np.random.randn(*(X_train.shape[1], 1))*0.01
numb_vectors = X_train.shape[0]
h = np.zeros(X_train.shape[0])
eta = 0.1
def Sigmoid(x):
    sig = 1/(1 + np.exp(-x))
    return sig
    
m = 650
iterations = 10
random_index = np.random.randint(m)
#indices = np.arange(len(X_train[:, 0]))
#random_indices = np.random.choice(indices, size = len(X_train[:, 0]), replace = False)
X_batches = np.split(X_train, 100, axis = 0)
X_batches = np.array(X_batches)
Y_batches = np.split(Y_train, 100, axis = 0)
Y_batches = np.array(Y_batches)
#print(X_batches[0].shape, Y_batches[0].shape)

for iter in range(iterations):
    for numb_batches in range(100):
        h = np.dot(X_batches[numb_batches], weights)
        Y_pred = Sigmoid(h)
        gradient = np.dot(X_batches[numb_batches].T, (Y_pred - Y_batches[numb_batches]))
        weights = weights - eta*gradient
    














##### plot a few Ising states
#matplotlib inline
"""
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
