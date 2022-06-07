from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X, y, features


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        plt.scatter(X[:,i], y, marker='o',c="orange")
        plt.xlabel(features[i])
        plt.ylabel('House Prices')
    
    plt.tight_layout()
    plt.show()
    
    
def fit_regression(X, Y):
   
    #Solving Normal equation to calculate weights= (X_transpose X + Lambda I)^-1 X_transpose y
    
    #added regularization as linalg doesn't invert X_Transpose X for scaled data
    lam = 1e-5
    #Add a column vector Xo=1 to the feature matrix to introduce Bias
    X0 = np.ones((X.shape[0],1))
    X=np.concatenate((X0,X),axis=1)
    XTran = np.matrix.transpose(X) 
    I= np.identity((XTran.dot(X)).shape[0])
    w = np.linalg.solve((XTran.dot(X) + lam * I),XTran.dot(Y))
        
    return  w
    


def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))
    print("Number of Instances: {}".format(X.shape[0]))
    print("Number of Attributes/Features: {}".format(X.shape[1]))
    
    #Visualize the features
    visualize(X, y, features)
        
    #Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=0)
    
    # standardizing data as the units of features are different
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test=scaler.transform(X_test)
  
  
    # Fit regression model
    w = fit_regression(X_train, y_train)
    
    #Part e) Print table of features with their corresponding weights
    print('Feature \t Weights \n')
    for i,feature in enumerate(features):
        print('{:<10}{:<20}'.format(feature,w[i+1]))
 
    
    #Part f) Compute fitted values, MSE, etc.
    #Add a column vector Xo=1 to the x_test to introduce Bias
    X0 = np.ones((X_test.shape[0],1))
    X_test=np.concatenate((X0,X_test),axis=1)
    y_pred = X_test.dot(w)
    mse= np.square(y_test - y_pred).mean()
    print("\n MSE : {}".format(mse))
    
    #part g) Calculate two more error measurement metrics
    print("\nOther error measurement metrics for Test Set:")
    #calculate Mean Absolute Error
    mae= abs(y_test - y_pred).mean()
    print('Mean Absolute Error is {}'.format(mae))
    #calculate Root Mean Square Error
    rmse = np.sqrt(mse)
    print("RMSE value is {}".format(rmse))
    

    
if __name__ == "__main__":
    main()


