import tensorflow as tf
import tensorflow.keras as keras

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#loads data from the mnist dataset
#x_train & x_test are the pixel data of each image
#y_train & y_test are the label of the image (0-9)

# print(x_train[0])
# print(y_train[0])

import matplotlib.pyplot as plt

#plt.imshow(x_train[0], cmap=plt.cm.binary)
#plt.show()

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) #128 units, rectifier activation function
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) #10 units, softmax activation function
model.compile(optimizer='adam', #basic optomizer
              loss='sparse_categorical_crossentropy', #error calculator
              metrics=['accuracy']) #what to track

model.fit(x_train, y_train, epochs=3) #train the model 3 times

val_loss, val_acc = model.evaluate(x_test, y_test)
print('loss:',val_loss)
print('acc:',val_acc)

import h5py #necessary for save?

model.save('epic_num_reader.model') #saves the model
new_model = tf.keras.models.load_model('epic_num_reader.model')
    #loads the model as a new instance of the model
predictions = new_model.predict(x_test) #
print(predictions)

import numpy as np

print("prediction:",np.argmax(predictions[0]))
plt.imshow(x_test[0], cmap=plt.cm.binary)
plt.show()
