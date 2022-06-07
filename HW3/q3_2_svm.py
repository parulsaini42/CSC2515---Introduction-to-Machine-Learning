

'''
Question 3.2 Code Part C
Implementing SVM with GrdSearchCV

Question 3.3
Performance metrics and ROC curve for svm classifier

'''
import data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc

def encoding(train_labels,test_labels):
    

    train_labels_encoded = label_binarize(train_labels,classes=range(10))
    test_labels_encoded = label_binarize(test_labels, classes=range(10))
    return train_labels_encoded,test_labels_encoded
    
def roc(train_data, train_labels, test_data, test_labels,model):
    
    n_classes=10
    y_train,y_test = encoding(train_labels,test_labels)
    model = OneVsRestClassifier(model).fit(train_data,y_train)
    # return target scores which will be used by roc curve as input
    y_score = model.predict_proba(test_data)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    for i in range(n_classes):
        plt.plot(fpr[i],tpr[i],lw=2,label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC Curve for SVM')
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
    
    
        
def svm_classifier(train_data, train_labels, test_data, test_labels):
    
    #parameter range
    param_grid = {'C': [1, 10, 100],
                  'kernel': ['rbf','linear','sigmoid','poly'],
                  'gamma': [1, 0.1, 0.01]
                  }
    #define grid to fine tune parameter, 'ovr' is set for multi class classification in SVM
    grid = GridSearchCV(SVC(decision_function_shape='ovr',probability=True,  random_state = 0), param_grid,verbose = 1)
    grid.fit(train_data, train_labels)
    print('Best Parameters for SVM:',grid.best_params_)
    model = grid.best_estimator_
    # print prediction results
    predictions = model.predict(test_data)
    
    return predictions, model
    
def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data_from_zip('a3digits.zip', 'data')

    predictions, model = svm_classifier(train_data, train_labels, test_data, test_labels)
    
    
    '''
    Q3 : 3.3 Classifier comparison
    '''
    #Call Function to calculate metrics
    performance_metrics(test_labels,predictions)
    
    #Show ROC Curve
    roc(train_data, train_labels, test_data, test_labels,model)
              
    
    
if __name__ == '__main__':
    main()
    

