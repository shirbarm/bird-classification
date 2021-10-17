import pandas as pd
import numpy as np

###########################   Train    ###########################
dataframeTrain = pd.read_pickle('dataframeTrain.pkl')
dataframeTrain = pd.DataFrame(dataframeTrain)

MFBmat1 = dataframeTrain['spectorgam'].tolist()  # extracting the MFB tensor
train_size = len(MFBmat1)
#print("train_size " + str(train_size))
x_train = np.asarray(MFBmat1[:train_size]).astype('float32') # saving it as a numpy array

x_shape = x_train.shape
#print(x_shape)
x_train_for_FCN = x_train.reshape(x_shape[0], x_shape[1]*x_shape[2])
x_train_for_FCN.shape

###########################   Test    ###########################

dataframeTest = pd.read_pickle('dataframeTest.pkl')
dataframeTest = pd.DataFrame(dataframeTest)

MFBmat2 = dataframeTest['spectorgam'].tolist()  # extracting the MFB tensor
test_size = len(MFBmat2)
x_test = np.asarray(MFBmat2[:test_size]).astype('float32') # saving it as a numpy array

x_shape = x_test.shape
print(x_shape)
x_train_for_FCN = x_test.reshape(x_shape[0], x_shape[1]*x_shape[2])
x_train_for_FCN.shape
