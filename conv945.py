import numpy as np
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D

import matplotlib.pyplot as plt
(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

x_train = x_train/255.0
x_test = x_test/255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test,10)

model = Sequential()

model.add(Conv2D(32,(3, 3), strides=(1,1), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32,(3, 3), strides=(1,1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), strides=(1,1), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), strides=(1,1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.35))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#optimizer = RMSprop(lr = 0.001, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=256, epochs=200, validation_data=(x_test,y_test))

score = model.evaluate(x_test,y_test)
print("score ",score[0])
print("accuracy ",score[1])

model.save("mnist_cnn.h5")
