
'''
Question 3.2 Code Part C
Implementing MLP using Scikit learn MLP Classifier

Question 3.3
Performance metrics and ROC curve for MLP

'''

import data
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
np.random.seed(0)

def encoding(train_labels,test_labels):
    
    enc = OneHotEncoder()
    enc.fit(train_labels.reshape(-1,1))
    train_labels_encoded = enc.transform(train_labels.reshape(-1,1)).toarray()
    test_labels_encoded = enc.transform(test_labels.reshape(-1,1)).toarray()
    return train_labels_encoded , test_labels_encoded

def roc(train_data, train_labels, test_data, test_labels,model):
    
    n_classes=10
    y_train = train_labels
    y_test = test_labels
    model = model.fit(train_data,y_train)
    # return target scores which will be used by roc curve as input
    y_score = model.predict_proba(test_data)
    fpr = dict()
    tpr = dict()
    roc_auc=dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    for i in range(n_classes):
        plt.plot(fpr[i],tpr[i],lw=2,label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC Curve for MLP')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='lower right')
    plt.show()

def performance_metrics(test_labels,predictions):
    
    accuracy = accuracy_score(test_labels, predictions)
    precision =precision_score(test_labels, predictions, average='weighted')
    recall = recall_score(test_labels, predictions, average='weighted')
    f1_score = 2 * (precision * recall) / (precision + recall)
    mse = mean_squared_error(test_labels, predictions)
    matrix=confusion_matrix(test_labels,predictions)
    classification_rpt=classification_report(test_labels, predictions,output_dict=True)
    
    print("\nAccuracy on Test data {}".format(accuracy))
    print("Precision {}".format(precision))
    print("Recall {}".format(recall))
    print("F1_Score {}".format(f1_score))
    print("MSE {}".format(mse))
    
    #Calculating class wise metrics
    FP = matrix.sum(axis=0) - np.diag(matrix)
    FN = matrix.sum(axis=1) - np.diag(matrix)
    TP = np.diag(matrix)
    TN = matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    F1 = 2 * (P * R) / (P + R)
    
    print("\nClasswise Precision\n",P)
    print("\nClasswise Recall\n",R)
    print("\nClasswise F1 Score\n",F1)
    print("\nConfusion Matrix:\n",matrix)
    
def mlp_classifier(train_data, train_labels, test_data, test_labels,solver,act_fn,learning_rate):
    
    '''
    Implement MLP classifier using scikit learn - Input and Output layers are determined automatically 
    actiavtion : Activation function for the hidden layer
    alpha      : L2 penalty (regularization term) parameter.
    solver     : The solver for weight optimization
    tol        : Tolerance for the optimization. When the loss or score is not improving by at least tol 
                 for n_iter_no_change consecutive iterations, unless learning_rate is set to ‘adaptive’,
                 convergence is considered to be reached and training stops.
    learning rate init : The initial learning rate used. 
                         It controls the step-size in updating the weights.
                         Only used when solver=’sgd’ or ‘adam’
    max_iter   : Maximum number of iterations unless it converges (based on tol)
    verbose    : Means the text output describing the process
    '''
    #To analyse performance on different number of units in the hidden layers
    print('Training Network')
    acc_test= []
    hidden_layers = [20,30,40,50,60,70,80,90,100,110,120]
    

    network=[]
    for h1 in hidden_layers:
        i=1
        h2 = h1 - i*5
        mlp = MLPClassifier(hidden_layer_sizes=(h1,h2,), activation=act_fn, alpha=1e-4,
                        solver=solver,tol=1e-4, random_state=0,
                        learning_rate_init=learning_rate,max_iter=3000, verbose=False)
        mlp.fit(train_data,train_labels)
        predict_test= mlp.predict(test_data)
        acc_test.append(accuracy_score(test_labels, predict_test))
        network.append([mlp.n_features_in_,h1,h2,mlp.n_outputs_,solver,act_fn,learning_rate,mlp.loss_,accuracy_score(test_labels, predict_test)])
        i=i+1
    
    df=pd.DataFrame(network,columns=['input','hidden1','hidden2' ,'output','solver','act_fn','lr','loss','test acc'])
    print(df)
    
              

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data_from_zip('a3digits.zip', 'data')
    #encoding labels
    train_labels_encoded ,test_labels_encoded = encoding(train_labels,test_labels)
    
    #best model obtained
    mlp_classifier(train_data, train_labels_encoded , test_data, test_labels_encoded ,'adam','relu',0.01)
    
    
    #Other configurations tested - results in report for single hidden layer
    '''
    mlp_classifier(train_data, train_labels_encoded, test_data, test_labels_encoded ,'adam','relu',0.001)
    mlp_classifier(train_data, train_labels_encoded , test_data, test_labels_encoded ,'adam','logistic',0.01)
    mlp_classifier(train_data, train_labels_encoded , test_data, test_labels_encoded ,'adam','logistic',0.001)
    mlp_classifier(train_data, train_labels_encoded , test_data, test_labels_encoded ,'sgd','relu',0.01)
    mlp_classifier(train_data, train_labels_encoded , test_data, test_labels_encoded ,'sgd','relu',0.001)
    mlp_classifier(train_data, train_labels_encoded, test_data, test_labels_encoded ,'sgd','logistic',0.01)
    mlp_classifier(train_data, train_labels_encoded, test_data, test_labels_encoded ,'sgd','logistic',0.001)
    '''
   
    
    
    
    '''
    Q3 : 3.3 Classifier comparison
    '''
    #Selecting best model based on are analysis
    best_model=MLPClassifier(hidden_layer_sizes=(90,85), activation='relu', alpha=1e-4,
                        solver='adam',tol=1e-4, random_state=0,
                        learning_rate_init=0.01,max_iter=3000, verbose=False)
    best_model.fit(train_data,train_labels_encoded)
    predictions_encoded=best_model.predict(test_data)
    
    #Converting back to labels for passing to performance_metrics
    y_predictions = (np.argmax(predictions_encoded, axis=1)).reshape(-1, 1)  
    predictions = []
    for item in y_predictions:
        predictions.append(item[0])
    
    #Call Function to calculate metrics
    performance_metrics(test_labels,predictions)
    
    #Show ROC Curve
    roc(train_data, train_labels_encoded, test_data, test_labels_encoded ,best_model)

    
   
    
if __name__ == '__main__':
    main()
    




