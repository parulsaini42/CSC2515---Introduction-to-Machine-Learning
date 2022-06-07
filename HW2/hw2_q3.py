from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from sklearn.datasets import load_boston
from itertools import count
import math
import pandas as pd
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']
count=count(start=1)
k=5
idx = np.random.permutation(range(N))

#helper function
def l2(A, B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist
 
#to implement
def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    '''
    Given a test datum, it returns its prediction based on locally weighted regression

    Input: test_datum is a dx1 test vector(query points matrix)
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter(also called bandwidth)
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''

    #Calculate the l2 norm of x and x^(j) as || x-x^(j)||^2 for computation of a^(i)
    dist= l2(np.matrix.transpose(test_datum),x_train)
    power=(-1) * (dist/ (2.0 * (tau ** 2)))
    numerator= np.exp(power)  
    denominator= np.exp(logsumexp(power))
    #Since A is a diagonal matrix such that Aii =a^(i)
    a = numerator/denominator
    A=np.diag(np.reshape(a,x_train.shape[0]))
    
    
  
    # Since w* = (X^T.A.X + Lambda * I)^-1 * (X^T.A.Y) 
    XTran = np.matrix.transpose(x_train) 
    I= np.identity((XTran.dot(A).dot(x_train)).shape[0])
    w = np.linalg.solve((XTran.dot(A).dot(x_train) + lam * I),XTran.dot(A).dot(y_train))  
    
    #Given y_hat = x^T w*
    y_pred = np.matrix.transpose(test_datum).dot(w)
    
    return y_pred

#helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector        
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
 
    print('Calculating Loss for Validation Run {}'.format(next(count)))
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        predictions =  np.array([LRLS(x_test[i,:].reshape(d,1),x_train,y_train, tau) \
                        for i in range(N_test)])
        losses[j] = ((predictions.flatten()-y_test.flatten())**2).mean()
    return losses

#to implement
def run_k_fold(x, y, taus, k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    loss=[]
    #Step 1, shuffle data and split into k groups of equal sizes 
    total_samples = x.shape[0]
    fold_size = total_samples/k
    fold_list_indexes = [idx[i:i+math.ceil(fold_size)] for i in range(0, len(idx), math.ceil(fold_size))]
    for i in range(0,k):
        #initialising 
        val_x=np.zeros((1, x.shape[1]))
        val_y=np.array([[0]])
        train_x=np.zeros((1, x.shape[1]))
        train_y=np.array([[0]])
        
        #Obtain the data for validation
        fold_list_copy=fold_list_indexes.copy()
        for j in fold_list_copy[i]:
            val_x=np.vstack((val_x ,x[j]))
            val_y=np.vstack((val_y ,y[j]))
        del fold_list_copy[i]
        #Obtaining the k-1 folds
        train_indexes = [j for m in fold_list_copy for j in m]
        for j in train_indexes:
            train_x =np.vstack((train_x , x[j]))
            train_y =np.vstack((train_y ,y[j]))
        #Pass matrices after removing initialising array
        loss.append(run_on_fold(val_x[1 : , : ],val_y[1:],train_x[1 : , : ],train_y[1:],taus))
       
    
    #calculate average losses for validation data for different values of tau over k validation run
    loss = np.sum(loss, 0)
    loss = loss/k   

       
    return loss

if __name__ == "__main__":
    
    # In this exercise we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)
    losses= run_k_fold(x,y,taus,k=5)
    
    #A table representing avg loss for each tau value
    df=pd.DataFrame({'Choice of Tau': list(range(1,201)), 'Value of tau':taus , 'avg_loss' : losses})
    print(df)
    
    #plot average losses for each choice of tau over k validation run
    plt.plot(taus,losses)
    plt.xlabel('value of tau')
    plt.ylabel('avg loss on validation data over {} validation runs '.format(k))
    plt.show()
  
    
    print("min loss = {}".format(losses.min()))
    
    
    
    
    


