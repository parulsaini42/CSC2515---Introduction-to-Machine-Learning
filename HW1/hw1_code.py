#Code written by Parul Saini
#Collaborators: Aditya Kharosekar, Subhayan Roy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn import tree

def load_data():
    #Creating a labelled DataFrame using the txt files
    news=[]
    with open('clean_real.txt') as file:
        for line in file:
            headline= line.rstrip("\n")
            news.append([headline,1])
    with open('clean_fake.txt') as file:
        for line in file:
            headline= line.rstrip("\n")
            news.append([headline,0])
            
    df= pd.DataFrame(news,columns=['headline','label'])
    y = df.label
    
    #Initialize CountVectorizer object - Assumption full vocabulary known at training time
    count_vectorizer = CountVectorizer(stop_words='english')
    vectorized_data = count_vectorizer.fit_transform(df['headline']).toarray()
    
    #Splitting data 
    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15
    x_train, x_test, y_train, y_test = train_test_split(vectorized_data, y, test_size=1 - train_ratio,random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + val_ratio),random_state=42) 
    
    print('Output of function load_data():\n')
    total_rec=df['headline'].count()
    print('Total Record Count:', total_rec)
    print('% Record Count for Training:', round((len(x_train)/total_rec)*100,0),'% = ',x_train.shape[0])
    print('% Record Count for Test:', round((len(x_test)/total_rec)*100,0),'% =',x_test.shape[0])
    print('% Record Count for Validation:', round((len(x_val)/total_rec)*100,0),'% =',x_val.shape[0])


    return x_train,y_train,x_test,y_test,x_val,y_val,count_vectorizer
    
     
    
def select_tree_model(x_train,y_train,x_test,y_test,x_val,y_val,vector):
    
    depth=[20,30,50,70,90]
    accuracy=[]
    
    #Analysing Decision Tree on validation data using two different split criteria (Information Gain and Gini coefficient)
    for m in depth:
        clf_e = DecisionTreeClassifier(criterion="entropy", max_depth=m,random_state=42)
        clf_e =  clf_e.fit(x_train,y_train)
        y_pred_e= clf_e.predict(x_val)
        accuracy.append(['entropy',m,round(metrics.accuracy_score(y_val, y_pred_e)*100,1)])
        
        clf_g = DecisionTreeClassifier(criterion="gini", max_depth=m,random_state=42)
        clf_g =  clf_g.fit(x_train,y_train)
        y_pred_g= clf_g.predict(x_val)
        accuracy.append(['gini',m,round(metrics.accuracy_score(y_val, y_pred_g)*100,1)])
    
    print('\nOutput of function select_tree_model():')
    #Printing accuracies of each model
    df_hyperparameters=pd.DataFrame(accuracy,columns=['Criteria','Max Depth','Accuracy'])
    print('\nHyperparameters and their corresponding accuracies for Decision Trees: \n', df_hyperparameters)
    
    
    #Hyperparameters with highlest validation accuracy:
    best_id=df_hyperparameters.index[df_hyperparameters['Accuracy'].idxmax()]
    criteria=df_hyperparameters['Criteria'][best_id]
    max_depth= df_hyperparameters['Max Depth'][best_id] 
    print('Hyperparameters with highest accuracy for Validation Set: Split Criteria: {} , Depth:{}'.format(criteria,max_depth))
    
    #Analysing model on best hyperparameters and test data
    clf = DecisionTreeClassifier(criterion=criteria, max_depth=max_depth,random_state=42)
    clf =  clf.fit(x_train,y_train)
    y_pred= clf.predict(x_test)
    print('Accuracy for DecisionTree, Test Set: ', round(metrics.accuracy_score(y_test, y_pred)*100,1) , '%')
    
    #Plot Decision Tree upto depth 2
    print('\nDecision Tree extract of first two layers:')
    feature_list=list(sorted(vector.vocabulary_.items(), key=lambda item: item[1]))
    plt.figure(figsize=(10,10))
    tree.plot_tree(clf, max_depth=2, fontsize=10,filled=True,rounded=True,class_names=['Fake','Real'],feature_names=feature_list)
    plt.show()
    

def compute_information_gain(x_train,y_train,vector,w):
    
    word = w
    y_train = list(y_train)
    total_headlines= x_train.shape[0]
    real_headlines = sum(y_train)
    fake_headlines = total_headlines - real_headlines
    
    ct_word_exists_real = 0
    ct_word_exists_fake = 0
    ct_word_not_exists_real = 0
    ct_word_not_exists_fake = 0
 
    #entropy before split:
    p_fake= fake_headlines/total_headlines
    p_real= real_headlines/total_headlines
    h_before_split = - p_fake*np.log2(p_fake) - p_real*np.log2(p_real)
    
    #Split criteria - real or fake headline | word exists or not 
    word_index = sorted(vector.vocabulary_).index(word)
  
    for i in range(total_headlines):
        if x_train[i][word_index] ==1:
            if y_train[i]==1:
                ct_word_exists_real +=1
            elif y_train[i]==0:
                ct_word_exists_fake +=1
        elif x_train[i][word_index] !=1:
            if y_train[i]==1:
                ct_word_not_exists_real +=1
            elif y_train[i]==0:
                ct_word_not_exists_fake +=1
    
    total_headlines_word_exists = ct_word_exists_real  + ct_word_exists_fake
    total_headlines_word_not_exists = ct_word_not_exists_real + ct_word_not_exists_fake
    
    p_word_exists_real = ct_word_exists_real/total_headlines_word_exists
    p_word_exists_fake = ct_word_exists_fake/total_headlines_word_exists
    p_word_not_exists_real = ct_word_not_exists_real/total_headlines_word_not_exists
    p_word_not_exists_fake = ct_word_not_exists_fake/total_headlines_word_not_exists
    
    h_left = - p_word_exists_real*np.log2(p_word_exists_real) - p_word_exists_fake*np.log2(p_word_exists_fake)
    h_right = - p_word_not_exists_real*np.log2(p_word_not_exists_real) - p_word_not_exists_fake*np.log2(p_word_not_exists_fake)
    
    #weighted entropy after split
    h_after_split = (total_headlines_word_exists/total_headlines)*h_left + (total_headlines_word_not_exists/total_headlines)*h_right
    IG = h_before_split - h_after_split
    
    print ('Information Gain for the word "{}" is {}'.format(word, IG)) 
   
        
def select_knn_model(x_train,y_train,x_test,y_test,x_val,y_val):
    
    error_train=[]
    error_val=[]
    accuracy_val=[]
    k_val= []
    N = x_train.shape[0]
    deg_freedom = []
    
    #Analysing training & validation error
    for k in range(1,21):
        clf_knn = KNeighborsClassifier(n_neighbors = k)
        clf_knn = clf_knn.fit(x_train,y_train)
        y_pred_train = clf_knn.predict(x_train)
        y_pred_val = clf_knn.predict(x_val)
        error_train.append(np.mean(y_pred_train != y_train))
        error_val.append(np.mean(y_pred_val != y_val))
        accuracy_val.append(round(metrics.accuracy_score(y_val, y_pred_val)*100,1))
        k_val.append(k)
        deg_freedom.append(round(N/k))
    
    
    #k corresponding to highlest validation accuracy (Note: index 0 is k=1 therfore add 1):
    best_k=accuracy_val.index(max(accuracy_val)) + 1
    print('Max Validation Accuracy --> Best k:', best_k)
    
    #Least validation error - best k
    min_val_error = min(error_val) 
    best_k = error_val.index(min_val_error) + 1
    print('Min Validation Error from Graph --> Best k:', best_k)
    
    #Analysing model on test data with best k
    clf_knn = KNeighborsClassifier(n_neighbors = best_k)
    clf_knn = clf_knn.fit(x_train,y_train)
    y_pred = clf_knn.predict(x_test)
    print('Accuracy for KNN with best k, Test Set: ', round(metrics.accuracy_score(y_test, y_pred)*100,1) , '%')
    
    
    #Plotting training and validation errors for values of k in the range 1 & 20
    deg_freedom.reverse()
    error_train.reverse()
    error_val.reverse()
    xticks = range(1,21,5)
    fig, ax = plt.subplots()
    ax.plot(deg_freedom,error_train, color='red',label='Training Error',marker='o',markerfacecolor='red', markersize=3)
    ax.plot(deg_freedom,error_val, color='blue',label='Validation Error',marker='o',markerfacecolor='blue', markersize=3)
    ax.set_xlabel('N/k')
    ax.set_ylabel('Error')
    ax.set_xticks(range(100,2500,400))
    ax.legend()
    ax2 = ax.twiny()
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticks[::-1])
    ax2.set_xlabel('k')
    plt.title('Missclassification curves for varying K values')
    plt.show()
    
    
def main():
    
    #Part a) Data Loading & Pre-processing
    x_train,y_train,x_test,y_test,x_val,y_val,vector=load_data()
    
    #Part b) & c) Function call to analyse Decision Tree -  
    select_tree_model(x_train,y_train,x_test,y_test,x_val,y_val,vector)
    
    
    #Part d)
    print('\nOutput of function compute_information_gain():\n')
    for w in ['donald','hillary','trumps','coal','election']:
        compute_information_gain(x_train,y_train,vector,w)
    
    #Part e) Function call to analyse KNN 
    print('\nOutput of function select_knn_model():\n')
    select_knn_model(x_train,y_train,x_test,y_test,x_val,y_val)
    
if __name__== "__main__":
    main()
    

