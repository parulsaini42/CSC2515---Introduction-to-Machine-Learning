'''
Question 3.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
np.random.seed(0)

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels
        

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        distance = self.l2_distance(test_point)
        #Sorting in ascending order while keeping the order of the indices. 
        #Select the first k nearest neighbours
        dist = np.argsort(np.array(distance))[:k]

        #Retrive the lables of the k nearest points
        labels = self.train_labels[dist]
  
        #Select and return the most frequently occuring label
        neighbors = {}

        for l in labels:
            if l in neighbors:
                neighbors[l] += 1
            else:
                neighbors[l] = 1

        # store label with maximum votes as the prediction for the point
        digit = max(neighbors, key=neighbors.get)
        
        return digit
    

    

def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    #no of folds
    k_cv=10
    
    #Code from Assignment 2
    total_samples = train_data.shape[0]
    fold_size = total_samples/k_cv
    idx = np.random.permutation(range(train_data.shape[0]))
    fold_list_indexes = [idx[i:i+math.ceil(fold_size)] for i in range(0, len(idx), math.ceil(fold_size))]
    avg_val_acc_folds=[]
    avg_train_acc_folds=[]
    for k in k_range:
        print('Making predictions when k={}'.format(k))
        # Loop over folds
        acc_val_folds=[]
        acc_train_folds=[]
        for i in range(0,k_cv):
            #initialising 
            val_x=np.zeros((1, train_data.shape[1]))
            val_y=np.array([[0]])
            train_x=np.zeros((1, train_data.shape[1]))
            train_y=np.array([[0]])

            #Obtain the data for validation
            fold_list_copy=fold_list_indexes.copy()
            for j in fold_list_copy[i]:
                val_x=np.vstack((val_x ,train_data[j]))
                val_y=np.vstack((val_y ,train_labels[j]))
            del fold_list_copy[i]
            #Obtaining the k-1 folds
            train_indexes = [j for m in fold_list_copy for j in m]
            for j in train_indexes:
                train_x =np.vstack((train_x , train_data[j]))
                train_y =np.vstack((train_y , train_labels[j]))
            
            #Since we had list of lists
            flat_train_y = [item for sublist in train_y[1:] for item in sublist]
            flat_val_y = [item for sublist in val_y[1:] for item in sublist]
            
            # Evaluate k-NN
            knn_cv = KNearestNeighbor(train_x[1 : , : ],np.array(flat_train_y))
            acc_val_folds.append(classification_accuracy(knn_cv,k,val_x[1 : , : ],np.array(flat_val_y)))
            acc_train_folds.append(classification_accuracy(knn_cv,k,train_x[1 : , : ],np.array(flat_train_y)))
        #calculate average accuracy for validation data for k_cv validation run
        avg_val_acc_folds.append([k,np.sum( acc_val_folds)/ k_cv])
        avg_train_acc_folds.append([k,np.sum( acc_train_folds)/ k_cv])
    
    
    #table representing k and corresponding avg accuracy in crossvalidation
    df1=pd.DataFrame(avg_val_acc_folds,columns=['k','avg_val_accuracy_cv'])
    print(df1)
    
    #table representing k and corresponding avg accuracy in crossvalidation
    df2=pd.DataFrame(avg_train_acc_folds,columns=['k','avg_train_accuracy_cv'])
    print(df2)
    
    return df1['k'][df1['avg_val_accuracy_cv'].idxmax()]
    
def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    predicted_label=[]
    for point in eval_data: 
        predicted_label.append(knn.query_knn(point, k))
        
    accuracy = accuracy_score(eval_labels,predicted_label)
    return accuracy
    

def main():
    
    train_data, train_labels, test_data, test_labels = data.load_all_data_from_zip('a3digits.zip', 'data')
    knn = KNearestNeighbor(train_data, train_labels)
    
   
    # Predict
    k_list=[1,15]
    for k in k_list:
        print('Making predictions when k={}'.format(k))
        acc_train = classification_accuracy(knn,k,train_data, train_labels)
        print('knn classification accuracy for train is {}'.format(acc_train) )
        
        acc_test = classification_accuracy(knn,k,test_data, test_labels)
        print('knn classification accuracy for test is {}'.format(acc_test) )
    
    #Perfom cross validation - best k has highest validation accuracy
    best_k=cross_validation(train_data, train_labels)
    print('best value of k is {}'.format(best_k))
    
    #best k test accuracy:
    acc_test_best = classification_accuracy(knn,best_k,test_data, test_labels)
    print('knn classification accuracy for test is for best value of k is {}'.format(acc_test_best) )
  
    
    
if __name__ == '__main__':
    main()


# In[ ]:




