'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import scipy.stats
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
np.random.seed(0)

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    for i in range(0, 10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        # Compute mean of class i
        mean_i=np.sum(i_digits,axis=0) / i_digits.shape[0]
        means[i]=   means[i] + mean_i

    # Compute means
    return means

def compute_sigma_mles(train_data, train_labels,means):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
   
    dim = 64
    covariances = np.zeros((10, 64, 64))
    for i in range(0, 10):
        class_mean = means[i]
        class_data = train_data[ train_labels == i] 
        train_diff = class_data-class_mean
        n=class_data.shape[0]
        # covariance = (x-mu)(x-mu)^T / (n-1) where n is the number of samples
        covariances[i] = covariances[i] + np.matmul(train_diff.T, train_diff)/(n-1) 
        + 0.01*np.eye(dim)
    return covariances

def plot_cov_diagonal(covariances):
    # Plot the log-diagonal of each covariance matrix side by side
    variances=[]
    for i in range(10):
        cov_diag = np.diag(covariances[i])
        # ...log of diagonal of covariance matrix for each class
        variances.append(np.log(cov_diag).reshape(8, 8))  
       
    
    # Plot all log of diagonals on same axis
    all_concat = np.concatenate(variances, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()
    
def multivariate_gaussian(mean, covariances,inv_cov_matrix,det_cov_matrix,x, param):
    '''
    Using formula for gaussina probability density we calculate 
    p(x|y=j,theta) for each class j
    '''
    d = 64
    # x_diff is 1xd
    x_diff= x - mean
    #x is 1xd , reshape to dx1 fot dot product as coavriance is dxd
    x=x.T
    #Using p(x|y) defined for gaussian MLE in the question
    pxy = np.exp(-0.5*np.dot(np.dot(x_diff.T,inv_cov_matrix), x))/np.sqrt(((2*np.pi)**d)*det_cov_matrix)
    logpxy = -0.5*d*np.log(2*np.pi) -0.5*np.log(det_cov_matrix) -0.5*np.dot(np.dot(x_diff.T,inv_cov_matrix), x)
    if param=='log':
        return logpxy
    else:
        return pxy
def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array - 7000 x 10 in our case
    '''
    
    n=digits.shape[0] 
    n_classes = 10
    #initialise
    logx_y = np.zeros((n,n_classes))
    #determinant is scalar value correpsonding to each of the 10 covariance matrices
    det_cov_matrix = np.zeros(covariances.shape[0])
    inv_cov_matrix = np.zeros((10, 64, 64)) 
    #Calculate inverse and determinant for covariance matrix for class i
    for j in range(0, covariances.shape[0]):
        det_cov_matrix[j] = np.linalg.det(covariances[j])
        inv_cov_matrix[j] = np.linalg.inv(covariances[j])
    for i in range(0, n):
        for j in range(0, n_classes):
            logx_y[i][j] = multivariate_gaussian(means[j], covariances[j], 
                    inv_cov_matrix[j], det_cov_matrix[j],digits[i], 'log') 
    return logx_y

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:
        log p(y|x, mu, Sigma)
    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    n_classes = 10
    n=digits.shape[0]
    '''
    Using bayes rule, p(y|x) = p(x|y)p(y)/p(x) Therefore log p(y|x) = log p(x|y) + log p(y) - log p(x)
    Since p(x) = Sum_k=1_10(p(y=k) product_j=1_to_N(p(xj|y=k)))
    '''
    #initialise
    logy_x=np.zeros((n,n_classes))
    logx_y = generative_likelihood(digits, means, covariances)
    log_x= np.zeros(n)
    # 1xk vector where k is the number of classes, k-10 in our case - given as prior
    p_y =[0.1 for i in range(0,10)]
    #p(x|y) is required for computation of log p(x)
    px_y = np.zeros((n,n_classes))
    inv_cov_matrix = np.zeros((10, 64, 64))
    #determinant is scalar value correpsonding to each of the 10 covariance matrices
    det_cov_matrix = np.zeros(covariances.shape[0])
    #Calculate inverse and determinant for covariance matrix for class i
    for j in range(0, covariances.shape[0]):
        det_cov_matrix[j] = np.linalg.det(covariances[j])
        inv_cov_matrix[j] = np.linalg.inv(covariances[j])
    #Calculate p(x|y)
    for i in range(0, n):
        for j in range(0, n_classes):
            px_y[i][j] = multivariate_gaussian( means[j], covariances[j], inv_cov_matrix[j],
                                               det_cov_matrix[j],digits[i],'pxy') 
    #We can calculate log p(x) as
    for i in range(0,n):
            log_x[i] = np.log(np.dot(px_y[i],p_y))
    
    #We can calculate log p(y|x) = log p(x|y) + log p(y) - log p(x)
    for i in range(0, n):
        for j in range(0, n_classes):
            logy_x[i][j]= logx_y[i][j] + np.log(p_y[j]) - log_x[i]
    return logy_x

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute as described above and return
    n=digits.shape[0]
    sum_likelihood= 0
    for i in range(0,n):
        #fetch the true class
        y_true=int(labels[i])
        sum_likelihood= sum_likelihood + cond_likelihood[i][y_true]
    avg_likelihood= float(sum_likelihood)/float(n)
    return avg_likelihood

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    n=digits.shape[0]
    #assuming size is same as labels
    y_pred=np.zeros(digits.shape[0])
    
    for i in range(0,n):
            #Calculated based on maximum conditional likelihood
            y_pred[i]=float(np.argmax(cond_likelihood[i]))
            
    return y_pred

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels,means)
    #Plot diagonal elements of each covariance matrix
    plot_cov_diagonal(covariances)
    
    #Calculate avergae conditional log likelihood on train and test data
    avg_train=avg_conditional_likelihood(train_data, train_labels, means, covariances)
    avg_test=avg_conditional_likelihood(test_data, test_labels, means, covariances)
    print('Avergae conditional log likelihood on train data is',avg_train)
    print('Avergae conditional log likelihood on test data is',avg_test)

    #Classify data
    y_train_pred = classify_data(train_data, means, covariances)
    y_test_pred = classify_data(test_data, means, covariances)
    
    # Evaluation
    acc_train= accuracy_score(train_labels,y_train_pred)
    acc_test= accuracy_score(test_labels,y_test_pred)
    print('Accuracy on train data is {} %'.format(acc_train))
    print('Accuracy on test data is {} %'.format(acc_test))

if __name__ == '__main__':
    main()


# In[ ]:




