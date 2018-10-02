import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import pickle

pickle_in = open("X.pickle", "rb") #open file with pickle, read in binary
X = pickle.load(pickle_in) #reads file's content as a pickle data stream

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

X = X/255.0 #dividing by 255... why?

model = Sequential() #using a sequential model
#Convolutional Neural Net (CNN)
#Convulution (taking data and creating map) -> Pooling (down sampling the map)
#-> Convolution -> Pooling -> ...
#-> Fully Connected Layer -> Output
model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:])) #adding
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) #converts 3d map features to 2d

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

model.fit(X, y, batch_size = 32, epochs = 3, validation_split = 0.3)
