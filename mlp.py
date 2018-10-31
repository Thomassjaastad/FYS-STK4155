
import numpy as np

np.random.seed(1)

class mlp():
    def __init__(self, inputs, targets, nhidden):
        self.inputs = inputs
        self.targets = targets
        self.nhidden = nhidden
        self.eta = 0.1
        self.beta = 1.0
        self.nvectors = inputs.shape[0]
        self.ntargets = targets.shape[1]
        self.ninputs = inputs.shape[1]
        self.hiddenacc = np.zeros(self.nhidden)
        self.output = np.zeros(targets.shape[1])
        self.v = np.random.randn(*(self.ninputs + 1, self.nhidden))*0.01
        self.w = np.random.randn(*(self.nhidden + 1, self.ntargets))*0.01

    # activation functions
    def sigmoid(self, x):
        return 1./(1 + np.exp(-self.beta*x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        a= 0.01
        if x < 0:
            return a*x
        else:
            return x
        #return 0 if x < 0 else x

    def include_bias(self, array):
        bias = -1
        return np.append(array, bias)

    def forward(self, inputs):
        """
        Takes in a single vector. 
        """
        inputs_tot = self.include_bias(inputs)
        h_chi = np.zeros(self.nhidden)
        a_chi = np.zeros(self.nhidden)
        h_kappa = np.zeros(self.ntargets)
        y_kappa = np.zeros(self.ntargets)
        
        # activation on 1st layer  
        h_chi = np.dot(inputs_tot, self.v)
        a_chi = self.sigmoid(h_chi)
        self.hiddenacc = a_chi                     
        
        # activation on 2nd layer
        a_chi_tot = self.include_bias(a_chi)   
        h_kappa = np.dot(a_chi_tot, self.w)
        
        # output
        y_kappa = self.relu(h_kappa)
        #print(y_kappa)
        #y_kappa = self.sigmoid(h_kappa)
        self.output = y_kappa
        return y_kappa

    def backward(self, inputs, targets):
        updateV = np.zeros(np.shape(self.v))
        updateW = np.zeros(np.shape(self.w))

        delO = np.zeros(self.ntargets)
        delH = np.zeros(self.nhidden)

        #Error in outputlayer
        #for k in range(self.ntargets):
        delO = (self.output - targets)#*self.sigmoid_derivative(self.output[k])                
        updateW = -self.eta*np.outer(self.hiddenacc.T, delO)
        #Error in hiddenlayer
        for k in range(self.nhidden):            
            delH[k] = self.hiddenacc[k]*(1.0 - self.hiddenacc[k])*(sum(delO[:]*self.w.T[:, k]))                 
        updateV = - self.eta*np.outer(inputs.T, delH)
        
        #updateV and updateW are one smaller than self.v and self.w because of bias.
        updateV = np.vstack((updateV, np.zeros(self.nhidden)))
        updateW = np.vstack((updateW, np.zeros(self.ntargets)))
        self.v += updateV
        self.w += updateW
        #for i in range(self.ninputs):
        #    self.v[i] = self.v[i] + updateV[i]
        #for j in range(self.nhidden):
        #    self.w[j] = self.w[j] + updateW[j]

    def train(self, inputs, targets):
        for n in range(inputs.shape[0]):
            self.forward(inputs[n])
            self.backward(inputs[n], targets[n])

    def error(self, validationset, validationstargets):
        error = np.zeros(validationset.shape[0])
        
        for i in range(validationstargets.shape[0]):
            validation_out = self.forward(validationset[i])
            #print(validation_out, validationstargets[i])
            error[i] = np.linalg.norm(validation_out - validationstargets[i])**2
            #print(error[i], validation_out, validationstargets)
        #exit()
        print(sum(error))
        return sum(error)

    def earlystopping(self, inputs, targets, validationset, validationstargets):
        epochs = 400
        count = 0
        error = np.zeros(epochs)
        epochs_final = 0
        for i in range(epochs - 1):
            self.train(inputs, targets)
            error[i] = self.error(validationset, validationstargets)       
            if error[i - 1] < error[i]:
                count += 1
            else:
                count = 0

            if count == 10:
                print('Error increasing %d times in a row. STOP' % count)
                print('Final epoch is:', i)
                epochs_final = i
                indices = np.linspace(i + 1, epochs, epochs - i)
                error = np.delete(error, indices)
                break
        return error, epochs_final

    

    def k_fold(self, inputs_total, targets_total, Numbfolds):
        dataset = np.split(inputs_total[:-7], Numbfolds)
        dataset = np.array(dataset)
        targetset = np.split(targets_total[:-7], Numbfolds)
        targetset = np.array(targetset)
        # Holding out test data:
        test = dataset[0]
        test_target = targetset[0]
        
        for i in range(dataset.shape[1]*Numbfolds, inputs_total.shape[0]):
            test = np.vstack((test, inputs_total[i,:]))
            test_target = np.vstack((test_target, targets_total[i,:]))
        #TRAINING AND VALIDATIONSET REMAIN, K-1 folds:
        train_and_test = np.delete(dataset, 0, 0)
        train_and_test_targets = np.delete(targetset, 0, 0)
        #print(train_and_test.shape)
        #print(train_and_test_targets.shape, np.delete(train_and_test_targets, 0 , 0).shape )
        valid = []
        train = []
        valid_targets = []
        train_targets = []
        for k in range(Numbfolds - 1):
            valid.append(train_and_test[k])
            valid_targets.append(train_and_test_targets[k])

            train.append(np.delete(train_and_test, k, 0))
            train_targets.append(np.delete(train_and_test_targets, k , 0))

            train_and_test = train_and_test
            train_and_test_targets = train_and_test_targets
           
        valid = np.array(valid)
        valid_targets = np.array(valid_targets)
        
        train = np.array(train) 
        train_targets = np.array(train_targets)
        train = np.reshape(train, (Numbfolds - 1,(Numbfolds - 2)*train_and_test.shape[1], 41))
        train_targets = np.reshape(train_targets, (Numbfolds-1, (Numbfolds - 2)*train_and_test.shape[1], 8))
        return train, train_targets, valid, valid_targets, test, test_target