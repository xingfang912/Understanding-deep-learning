# Understanding-deep-learning
Python version: 2.7.X

Library dependencies: Numpy, OpenCV, theano, lasagne

CNN_library.py contains the ConvNet used in this research. The network is named as the Tiny VGG net.

DECNN_library.py contains the deconvolutional network of the Tiny VGG, namely Deconv_Tiny_VGG.

rafD_data_processing.py is the script for processing the data. The data can be acquired from: http://www.socsci.ru.nl:8180/RaFD2/RaFD?p=main

train.py is the script for training the network. Once the training is done, the parameters of the trained network along with the training/testing data and their labels, the ground truth labels, and the miss-classified cases are all saved under the current working directory.

test.py is for loading up a trained network and performing the testing.

deconv_Tiny_VGG.py contians the implementation of the backtracking algorithm and the feature tracking algorithm.
