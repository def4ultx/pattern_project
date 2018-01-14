import keras
import pandas as pd
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
nb_epoch = 20 #Number of times the whole data is used to learn


data = pd.read_csv('mnist_train.csv',header=None)
data = data.apply(pd.to_numeric)
data_array = data.as_matrix()
X, X_test, y, y_test = train_test_split(data_array[:,0:784],data_array[:,-1],test_size=0.2)

 
#Make the value floats in [0;1] instead of int in [0;255]
X= X.astype('float32')
X_test = X_test.astype('float32')
X /= 255
X_test /= 255
 
#Display the shapes to check if everything's ok
print(X.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
 
# convert class vectors to binary class matrices (ie one-hot vectors)
y = np_utils.to_categorical(y)
Y_test = np_utils.to_categorical(y_test)

#Define the model achitecture
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Dense(10)) #Last layer with one output per class
model.add(Activation('softmax')) #We want a score simlar to a probability for each class
 

rms = RMSprop()
#The function to optimize is the cross entropy between the true label and the output (softmax) of the model
model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=["accuracy"])
 
#Make the model learn
model.fit(X, y, nb_epoch=nb_epoch,
verbose=2,
validation_data=(X_test, Y_test))
 
#Evaluate how the model does on the test set
score = model.evaluate(X_test, Y_test, verbose=0)
 
print('Test score:', score[0])
print('Test accuracy:', score[1])