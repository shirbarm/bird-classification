# bird-classification
# Model 1 - Fully Connected:
In this network each neuron in one layer is connected to all the neurons in the next layer.

# Model 2 - Convolutional Neural Network
CNN is a type of neural network used primarily for artificial image analysis.
Our convolution network uses convolution layers and pooling layers. 

# Bird Voice Database 
479 train files and 201 test files.
Both groups contain 11 species of birds,
The length of the recordings ranges from one second to 19 seconds.
To simplify the work, we went over all the recordings and built a spectrogram for each recording. We found the median of the recordings: 0.58 seconds.
We then constructed a matrix from the spectrogram, and each bird was given a label between 1-11. We put all the data into the dataframe, and when we built the models we extracted the appropriate data from the dataframe.
