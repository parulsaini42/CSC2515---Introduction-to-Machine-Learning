'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
import random
import scipy.stats
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
np.random.seed(0)

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    #Since p(eta)=Beta(2,2)
    a=2
    b=2
    eta = np.zeros((10, 64))
    #Identity matrix of the shape of i_digits
    I=np.ones((700,64)) 
    for i in range(0, 10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        # Compute eta for each class i
        eta_i=(np.sum(i_digits,axis=0) + a - 1)/ (np.sum(i_digits,axis=0) + 
                ( np.sum(I,axis=0)- np.sum(i_digits,axis=0) )+ a + b - 2)
        eta[i]=   eta[i] + eta_i

    return eta

def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    
    img=[]
    for i in range(10):
        img_i = class_images[i]
        img.append(img_i.reshape(8, 8))
    
    all_concat = np.concatenate(img, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()
        

def likelihood(bin_digits,eta):
    
    #d=64
    d=len(bin_digits)
    # I is dX1 identity matrix
    I=np.ones(d)
    #reshaping for dot product - b is dX1
    b=bin_digits.T
    b_y=1
    for i in range(0, d):
        b_y= b_y * np.dot(np.power(eta,b[i]), np.power((I-eta),(1-b[i])).T)
    return b_y


def generate_new_data(bin_digits,eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    
    n_classes = 10
    n=bin_digits.shape[0]
    #initialise
    px_y = np.zeros((n,n_classes))
    for i in range(0, n):
        for j in range(0, n_classes):
            px_y[i][j]=likelihood(bin_digits[i],eta[j])
    
    generated_data = np.zeros((10, 64))   
    for i in range(0, 10):
        gen_xi = px_y[:][i]
        sample_i= random.choice(gen_xi)
        generated_data[i]=generated_data[i]+sample_i
    
    plot_images(generated_data)



def log_likelihood(bin_digits,eta):
    
    #d=64
    d=len(bin_digits)
    # I is dX1 identity matrix
    I=np.ones(d)
    #reshaping for dot product - b is dX1
    b=bin_digits.T
    logb_y= np.dot(b, np.log(eta)) + np.dot((I.T-b), np.log(I-eta))
    return logb_y
        
def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array 
    '''
    
    n_classes = 10
    n=bin_digits.shape[0]
    #initialise
    logx_y = np.zeros((n,n_classes))
    for i in range(0, n):
        for j in range(0, n_classes):
            logx_y[i][j]=log_likelihood(bin_digits[i],eta[j])
    return logx_y

def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    n_classes = 10
    n=bin_digits.shape[0]
    
    '''
    Using bayes rule, p(y|x) = p(x|y)p(y)/p(x)
    Therefore log p(y|x) = log p(x|y) + log p(y) - log p(x)
    we donot need to calculate the denominator as we are just calculating the likelihood.
    It will be done in a similar way as in the previous question.
    '''
     #initialise
    logy_x=np.zeros((n,n_classes))
    logx_y = generative_likelihood(bin_digits, eta)
    log_x= np.zeros(n)
    
    # 1xk vector where k is the number of classes, k-10 in our case - given as prior
    p_y =[0.1 for i in range(0,10)]
    
    for i in range(0, n):
        for j in range(0, n_classes):
            logy_x[i][j]= logx_y[i][j] + np.log(p_y[j]) 
    return logy_x


def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    # Compute as described above and return
    n=bin_digits.shape[0]
    sum_likelihood= 0
    for i in range(0,n):
        #fetch the true class
        y_true=int(labels[i])
        sum_likelihood= sum_likelihood + cond_likelihood[i][y_true]
    avg_likelihood= float(sum_likelihood)/float(n)
    return avg_likelihood


def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    # Compute and return the most likely class
    n=bin_digits.shape[0]
    #assuming size is same as labels
    y_pred=np.zeros(bin_digits.shape[0])
    
    for i in range(0,n):
           y_pred[i]=float(np.argmax(cond_likelihood[i]))
            
    return y_pred
   

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)
    print('Part 3) Plot of each of ηk vectors(eta) as an 8 by 8 grayscale image')
    plot_images(eta)


    #Genrate new data point from distribution
    print('Part 4) Plot of each of generated datapoint as an 8 by 8 grayscale image')
    generate_new_data(train_data,eta)
    
    
    #Calculate avergae conditional log likelihood on train and test data
    avg_train=avg_conditional_likelihood(train_data, train_labels, eta)
    avg_test=avg_conditional_likelihood(test_data, test_labels,eta)
    print('Avergae conditional log likelihood on train data is',avg_train)
    print('Avergae conditional log likelihood on test data is',avg_test)


    #Classify data
    y_train_pred = classify_data(train_data, eta)
    y_test_pred = classify_data(test_data, eta)
    
    # Evaluation
    acc_train= accuracy_score(train_labels,y_train_pred)
    acc_test= accuracy_score(test_labels,y_test_pred)
    print('Accuracy on train data is {} %'.format(acc_train))
    print('Accuracy on test data is {} %'.format(acc_test))
    
if __name__ == '__main__':
    main()

