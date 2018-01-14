import numpy as np
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.utils import np_utils
from keras.optimizers import RMSprop

from sklearn.model_selection import train_test_split
nb_epoch = 20 #Number of times the whole data is used to learn

a=50 #Number of component

import pandas as pd
data = pd.read_csv('mnist_train_sort.csv',header=None)

X = data.ix[:,1:28*28+1].values
y = data.ix[:,0].values

#Make the value floats in [0;1] instead of int in [0;255]
X= X.astype('float32')
X /= 255

pca = PCA(n_components=a)
pca.fit(X)
eig=pca.components_
# print(eig[1,:].shape)
# print(eig[:,1].shape)
X = np.dot(X, eig.T) # transform

# print(len(X_new))
# print(len(X_new[0]))

eig_vals=pca.explained_variance_
# X_inverse = np.dot(X_new, eig) # inverse_transform

#X, X_test, y, y_test = train_test_split(X_new,Y,test_size=0.2)

data_test = pd.read_csv('mnist_test_sort.csv',header=None)
X_test = data_test.ix[:,1:28*28+1].values
y_test = data_test.ix[:,0].values

X_test = X_test.astype('float32')
X_test /= 255

X_test = np.dot(X_test, eig.T)

img_rows, img_cols = a, 1

X = X.reshape(X.shape[0],  img_rows, img_cols,1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols,1)
input_shape = (img_rows, img_cols, 1)

#Display the shapes to check if everything's ok
print(X.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
 
# convert class vectors to binary class matrices (ie one-hot vectors)
y = np_utils.to_categorical(y)
Y_test = np_utils.to_categorical(y_test)

model = Sequential()
# Neural Network - Multilayer Perceptron
model.add(Flatten(input_shape=input_shape))
model.add(Dense(128, activation='relu',))
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
