import data
import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout
from sklearn.metrics import accuracy_score
np.random.seed(0)

if __name__ == '__main__':

    #We have 64 features and 10 target classes 
    train_data, train_labels, test_data, test_labels = data.load_all_data_from_zip('a3digits.zip', 'data')
    X_train = train_data.reshape(train_data.shape[0], 64)
    X_test = test_data.reshape(test_data.shape[0], 64)
    Y_train = to_categorical(train_labels, 10)
    Y_test = to_categorical(test_labels, 10)
    model = Sequential()
    #Hidden layer1 with 90 units and connected to an input layer of 64 units
    model.add(Dense(90, input_shape=(64,), activation='relu'))
    #Hidden layer2 with 85 units 
    model.add(Dense(85, activation='relu'))
    #Dropout is added to prevent overfitting
    model.add(Dropout(0.2))
    #Output layer with 10 units
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=30, batch_size=400, validation_split=0.3)
    y_pred1 = model.predict(X_test)
    #convert categorical values to label
    y_pred = np.argmax(y_pred1, axis=1)
    print('Accuracy:', accuracy_score(test_labels, y_pred))
    
    