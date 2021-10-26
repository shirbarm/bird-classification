# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 15:20:21 2021


"""
import librosa
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import os

inputs = keras.Input(shape=(35, 26, 1))
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)
outputs = layers.Dense(11, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

from tensorflow.keras.datasets import mnist

###########################   Train    ###########################
# Train Image
dataframeTrain = pd.read_pickle('dataframeTrain.pkl')
dataframeTrain = pd.DataFrame(dataframeTrain)

train_images = dataframeTrain['S_matriz'].tolist()  # extracting the MFB tensor
train_size = len(train_images)
x_train = np.asarray(train_images[:train_size]).astype('float32') # saving it as a numpy array

x_shape = x_train.shape
x_train_for_FCN = x_train.reshape(x_shape[0], x_shape[1], x_shape[2])
print(x_train_for_FCN.shape)

# Train Label
train_labels = dataframeTrain['label'].tolist()
train_size = len(train_labels)
y_train = np.asarray(train_labels[:train_size]).astype('float32')

y_shape = y_train.shape
y_train_for_FCN = y_train.reshape(y_shape[0])
print(y_train_for_FCN.shape)


###########################   Test    ###########################
# Test Image
dataframeTest = pd.read_pickle('dataframeTest.pkl')
dataframeTest = pd.DataFrame(dataframeTest)

test_images = dataframeTest['S_matriz'].tolist()  # extracting the MFB tensor
test_size = len(test_images)
x_test = np.asarray(test_images[:test_size]).astype('float32') # saving it as a numpy array

x_shape = x_test.shape
x_test_for_FCN = x_test.reshape(x_shape[0], x_shape[1], x_shape[2])
print(x_test_for_FCN.shape)

# Test Label
test_labels = dataframeTest['label'].tolist()
test_size = len(test_labels)
y_test = np.asarray(test_labels[:test_size]).astype('float32')

y_shape = y_test.shape
y_test_for_FCN = y_test.reshape(y_shape[0])
print(y_test_for_FCN.shape)

#######################################################################################
train_images = x_train_for_FCN.reshape((479, 35, 26, 1))
train_images = np.abs(x_train_for_FCN.astype('float32') / (-80))
test_images = x_test_for_FCN.reshape((201, 35, 26, 1))
test_images = np.abs(x_test_for_FCN.astype('float32') / (-80))

model.compile(optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])

model.fit(train_images, y_train_for_FCN, epochs=150, batch_size=32)
test_loss, test_acc = model.evaluate(test_images, y_test_for_FCN)
print(f"Test accuracy: {test_acc:.3f}")

print("The expected number is: 1 - Prinia gracilis")
tmp = test_images[1].reshape((1, 35, 26, 1))
pred = model.predict(tmp)
classes = np.argmax(pred, axis = 1)
classes[0]+=1
print(f"Test Bird is: " + str(classes))
print()

print("The expected number is: 2 - Parus major")
tmp = test_images[59].reshape((1, 35, 26, 1))
pred = model.predict(tmp)
classes = np.argmax(pred, axis = 1)
classes[0]+=1
print(f"Test Bird is: " + str(classes))
print()

print("The expected number is: 3 - Passer domesticus")
tmp = test_images[80].reshape((1, 35, 26, 1))
pred = model.predict(tmp)
classes = np.argmax(pred, axis = 1)
classes[0]+=1
print(f"Test Bird is: " + str(classes))
print()

print("The expected number is: 4 - Turdus merula")
tmp = test_images[97].reshape((1, 35, 26, 1))
pred = model.predict(tmp)
classes = np.argmax(pred, axis = 1)
classes[0]+=1
print(f"Test Bird is: " + str(classes))
print()

print("The expected number is: 5 - Chloris chloris")
tmp = test_images[110].reshape((1, 35, 26, 1))
pred = model.predict(tmp)
classes = np.argmax(pred, axis = 1)
classes[0]+=1
print(f"Test Bird is: " + str(classes))
print()

print("The expected number is: 6 - Carduelis carduelis")
tmp = test_images[115].reshape((1, 35, 26, 1))
pred = model.predict(tmp)
classes = np.argmax(pred, axis = 1)
classes[0]+=1
print(f"Test Bird is: " + str(classes))
print()

print("The expected number is: 7 - Cinnyris osea")
tmp = test_images[133].reshape((1, 35, 26, 1))
pred = model.predict(tmp)
classes = np.argmax(pred, axis = 1)
classes[0]+=1
print(f"Test Bird is: " + str(classes))
print()

print("The expected number is: 8 - Halcyon smyrnensis")
tmp = test_images[155].reshape((1, 35, 26, 1))
pred = model.predict(tmp)
classes = np.argmax(pred, axis = 1)
classes[0]+=1
print(f"Test Bird is: " + str(classes))
print()

print("The expected number is: 9 - Pycnonotus xanthopygos")
tmp = test_images[198].reshape((1, 35, 26, 1))
pred = model.predict(tmp)
classes = np.argmax(pred, axis = 1)
classes[0]+=1
print(f"Test Bird is: " + str(classes))
print()

print("The expected number is: 10 - Acrocephalus scirpaceus")
tmp = test_images[40].reshape((1, 35, 26, 1))
pred = model.predict(tmp)
classes = np.argmax(pred, axis = 1)
classes[0]+=1
print(f"Test Bird is: " + str(classes))
print()

print("The expected number is: 11 - Acrocephalus arundinaceus")
tmp = test_images[50].reshape((1, 35, 26, 1))
pred = model.predict(tmp)
classes = np.argmax(pred, axis = 1)
classes[0]+=1
print(f"Test Bird is: " + str(classes))
print()
