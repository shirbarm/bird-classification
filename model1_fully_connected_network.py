# -*- coding: utf-8 -*-
"""
Created on Sun May 16 12:50:20 2021

@author: user
"""
# Introduction

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import pandas as pd

###########################   Train    ###########################
# Train Image
dataframeTrain = pd.read_pickle('dataframeTrain.pkl')
dataframeTrain = pd.DataFrame(dataframeTrain)

train_images = dataframeTrain['spectorgam'].tolist()  # extracting the MFB tensor
train_size = len(train_images)
x_train = np.asarray(train_images[:train_size]).astype('float32') # saving it as a numpy array

x_shape = x_train.shape
x_train_for_FCN = x_train.reshape(x_shape[0], x_shape[1]*x_shape[2])
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

test_images = dataframeTest['spectorgam'].tolist()  # extracting the MFB tensor
test_size = len(test_images)
x_test = np.asarray(test_images[:test_size]).astype('float32') # saving it as a numpy array

x_shape = x_test.shape
x_test_for_FCN = x_test.reshape(x_shape[0], x_shape[1]*x_shape[2])
print(x_test_for_FCN.shape)

# Test Label
test_labels = dataframeTest['label'].tolist()
test_size = len(test_labels)
y_test = np.asarray(test_labels[:test_size]).astype('float32')

y_shape = y_test.shape
y_test_for_FCN = y_test.reshape(y_shape[0])
print(y_test_for_FCN.shape)

###########################   Train the Model    ###########################
for i in range(1):
    print(i)
    #plt.figure(1, figsize=(9, 3))
    #plt.imshow(y_train_for_FCN[i])
    #plt.suptitle('label =' + str(train_labels[i]))
    #plt.show()
    #print('label = ', train_labels.iloc[[i]])
    print('label = ', train_labels[i])
    plt.pause(0.1)

# The network architecture
from keras import models
from keras import layers

model = models.Sequential([
    layers.Dense(512, activation='relu'),   # input_shape=(479, 910)
    layers.Dense(11, activation='softmax')
])

# compilation step
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


train_images = np.abs(x_train_for_FCN.astype('float32') / (-80))
test_images = np.abs(x_test_for_FCN.astype('float32') / (-80))

from tensorflow.keras.utils import to_categorical

train_labels = to_categorical(y_train_for_FCN)
test_labels = to_categorical(y_test_for_FCN)

model.fit(train_images, train_labels, epochs=150, batch_size=32)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test accuracy =', test_acc)

print()
print()
print()
print()
print()
print(model.summary())
















