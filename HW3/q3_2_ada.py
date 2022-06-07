
'''
Question 3.2 Code Part C
Implementing AdaBoost classifier to turn a weak-learner into a strong performing classifier. 

Question 3.3
Performance metrics and ROC curve for adaboost classifier
'''
import data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
np.random.seed(0)

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
    plt.title('ROC Curve for AdaBoost')
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
    
    print("\nPerformance metrics for AdaBoostClassifier")
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
    
   
    
def adaboost(train_data, train_labels, test_data, test_labels):
    
    #Default base estimator for AdaBoost
    Weak_learner=DecisionTreeClassifier(max_depth=1)
    
    #Using GridSearchCV for finding the best parameters
    param_grid = {
                   'learning_rate':[0.01,0.1,1],
                   'n_estimators' : [50,100,150,200,300]

                  }
    print("Weak Learner Considered is {}".format(Weak_learner))
    classifier=AdaBoostClassifier(base_estimator= Weak_learner)
    grid = GridSearchCV(estimator=classifier, param_grid= param_grid, verbose =1)
    grid.fit(train_data, train_labels)
    print('\nBest Parameters are:',grid.best_params_)
    AdaBoost = grid.best_estimator_
    # print prediction results
    predictions = AdaBoost.predict(test_data)
    
    return predictions,AdaBoost
    
          
def main():
    
    train_data, train_labels, test_data, test_labels = data.load_all_data_from_zip('a3digits.zip', 'data')
          
    #Call classifier    
    predictions,best_model= adaboost(train_data, train_labels, test_data, test_labels)
    
    #performance of weak learner without boosting
    weak_learner = DecisionTreeClassifier(max_depth=1).fit(train_data, train_labels)
    weak_predictions= weak_learner.predict(test_data)
    print("\nAccuracy on Test data for Weak Learner {}".format(accuracy_score(test_labels,weak_predictions)))
    
    
    '''
    Q3 : 3.3 Classifier comparison
    '''
    #Call Function to calculate metrics
    performance_metrics(test_labels,predictions)
    
    #Show ROC Curve
    roc(train_data, train_labels, test_data, test_labels,best_model)
    
    
if __name__ == '__main__':
    main()
    





