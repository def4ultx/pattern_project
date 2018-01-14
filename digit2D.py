import keras
import pandas as pd
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
nb_epoch = 20 #Number of times the whole data is used to learn

train_data = pd.read_csv('mnist_train_sort.csv',header=None)
train_data = train_data.apply(pd.to_numeric)
data_array = train_data.as_matrix()
X = data_array[:,1:785]
y = data_array[:,0]

test_data = pd.read_csv('mnist_test_sort.csv',header=None)
test_data = test_data.apply(pd.to_numeric)
data_array = test_data.as_matrix()
X_test = data_array[:,1:785]
y_test = data_array[:,0]


# data = pd.read_csv('digit.csv',header=None)
# data = data.apply(pd.to_numeric)
# data_array = data.as_matrix()
# X, X_test, y, y_test = train_test_split(data_array[:,0:28*28],data_array[:,-1],test_size=0.2)

img_rows, img_cols = 28, 28

X = X.reshape(X.shape[0],  img_rows, img_cols,1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols,1)
input_shape = (img_rows, img_cols, 1)
 
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
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

rms = RMSprop()
#The function to optimize is the cross entropy between the true label and the output (softmax) of the model
model.compile(loss='categorical_crossentropy', optimizer= rms, metrics=["accuracy"])
 
#Make the model learn
model.fit(X, y, epochs=nb_epoch,
verbose=2,
validation_data=(X_test, Y_test))
 
#Evaluate how the model does on the test set
score = model.evaluate(X_test, Y_test, verbose=0)
 
print('Test score:', score[0])
print('Test accuracy:', score[1])
